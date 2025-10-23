import logging
import random
import asyncio
from pydantic import BaseModel
import litellm
from dspy.utils.exceptions import ParseError
from dspy.adapters.types.base_type import split_message_content_for_custom_types
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.clients.base_lm import BaseLM
from dspy.clients.lm import LM
from dspy.dsp.utils.settings import settings
from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.streaming.messages import StreamResponse
from dspy.signatures.signature import Signature, ensure_signature
from dspy.utils.callback import BaseCallback
from typing import Any, get_origin, List, AsyncIterator
from asyncio import Queue
from ..alto.lmtextpipe import LMTextPipe, WordOutput, LineOutput, SentenceOutput, FullTextOutput
#from anyio import create_memory_object_stream, create_task_group
logger = logging.getLogger(__name__)

class StreamPredict(Module):
    """Basic DSPy module that maps inputs to outputs using a language model.

    Args:
        signature: The input/output signature describing the task.
        callbacks: Optional list of callbacks for instrumentation.
        **config: Default keyword arguments forwarded to the underlying
            language model. These values can be overridden for a single
            invocation by passing a ``config`` dictionary when calling the
            module. For example::

                predict = dspy.Predict("q -> a", rollout_id=1, temperature=1.0)
                predict(q="What is 1 + 52?", config={"rollout_id": 2, "temperature": 1.0})
    """

    def __init__(self, signature: str | type[Signature], callbacks: list[BaseCallback] | None = None,
        is_streaming: bool = False, grandularity = FullTextOutput, **config):
        super().__init__(callbacks=callbacks)
        self.stage = random.randbytes(8).hex()
        self.signature = ensure_signature(signature)
        self.is_streaming = is_streaming
        #self._predict = Predict(signature, callbacks=callbacks, **config)
        self.config = config
        self.grandularity = grandularity
        self.reset()
    # def forward(self, **kwargs):

    def reset(self):
        self.lm = None
        self.traces = []
        self.train = []
        self.demos = []

    async def aforward(self, **kwargs):
        lm, config, signature, demos, inputs = self._forward_preprocess(**kwargs)
        prompt = self._format_messages(signature=signature, demos=demos,inputs=inputs)
        out_field_name, _ = next(iter(signature.output_fields.items()))
        if self.is_streaming:
            pred_stream = await self._lm_stream_caller(
            lm=lm,
            config=config,
            signature=signature,
            messages=prompt,
            out_field_name=out_field_name,
            )
            return pred_stream
        
        final_text, lmtextpipe_output = await self._lm_caller(
        lm=lm,
        config=config,
        signature=signature,
        messages=prompt,
        out_field_name=out_field_name,
        )
        completions = self._parse_messages(signature, final_text, lmtextpipe_output)

        pred = self.wrap_as_prediction(completions, signature)
        return pred

    
    def _forward_preprocess(self, **kwargs):
        # Extract the three privileged keyword arguments.
        assert "new_signature" not in kwargs, "new_signature is no longer a valid keyword argument."
        signature = ensure_signature(kwargs.pop("signature", self.signature))
        assert len(signature.output_fields) == 1, f"StreamPredict expects exactly 1 output field, got {len(signature.output_fields)}: {list(signature.output_fields.keys())}"
        demos = kwargs.pop("demos", None)
        config = {**self.config, **kwargs.pop("config", {})}

        # Get the right LM to use.
        lm = kwargs.pop("lm", self.lm) or settings.lm

        if lm is None:
            raise ValueError(
                "No LM is loaded. Please configure the LM using `dspy.configure(lm=dspy.LM(...))`. e.g, "
                "`dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))`"
            )

        if isinstance(lm, str):
            # Many users mistakenly use `dspy.configure(lm="openai/gpt-4o-mini")` instead of
            # `dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))`, so we are providing a specific error message.
            raise ValueError(
                f"LM must be an instance of `dspy.BaseLM`, not a string. Instead of using a string like "
                f"'dspy.configure(lm=\"{lm}\")', please configure the LM like 'dspy.configure(lm=dspy.LM(\"{lm}\"))'"
            )
        elif not isinstance(lm, BaseLM):
            raise ValueError(f"LM must be an instance of `dspy.BaseLM`, not {type(lm)}. Received `lm={lm}`.")

        # If temperature is unset or <=0.15, and n > 1, set temperature to 0.7 to keep randomness.
        temperature = config.get("temperature") or lm.kwargs.get("temperature")
        num_generations = config.get("n") or lm.kwargs.get("n") or lm.kwargs.get("num_generations") or 1

        if (temperature is None or temperature <= 0.15) and num_generations > 1:
            config["temperature"] = 0.7

        if "prediction" in kwargs:
            if (
                isinstance(kwargs["prediction"], dict)
                and kwargs["prediction"].get("type") == "content"
                and "content" in kwargs["prediction"]
            ):
                # If the `prediction` is the standard predicted outputs format
                # (https://platform.openai.com/docs/guides/predicted-outputs), we remove it from input kwargs and add it
                # to the lm kwargs.
                config["prediction"] = kwargs.pop("prediction")

        if not all(k in kwargs for k in signature.input_fields):
            present = [k for k in signature.input_fields if k in kwargs]
            missing = [k for k in signature.input_fields if k not in kwargs]
            logger.warning(
                "Not all input fields were provided to module. Present: %s. Missing: %s.",
                present,
                missing,
            )
        return lm, config, signature, demos, kwargs

    def _get_positional_args_error_message(self):
        input_fields = list(self.signature.input_fields.keys())
        return (
            "Positional arguments are not allowed when calling `dspy.StreamPredict`, must use keyword arguments "
            f"that match your signature input fields: '{', '.join(input_fields)}'. For example: "
            f"`predict({input_fields[0]}=input_value, ...)`."
        )
    
    # def __call__(self, *args, **kwargs):
    #     if args:
    #         raise ValueError(self._get_positional_args_error_message())

    #     return super().__call__(**kwargs)

    async def acall(self, *args, **kwargs):
        if args:
            raise ValueError(self._get_positional_args_error_message())

        return await super().acall(**kwargs)

    def dump_state(self, json_mode=True):
        state_keys = ["traces", "train"]
        state = {k: getattr(self, k) for k in state_keys}

        state["demos"] = []
        for demo in self.demos:
            demo = demo.copy()

            for field in demo:
                # FIXME: Saving BaseModels as strings in examples doesn't matter because you never re-access as an object
                demo[field] = serialize_object(demo[field])

            if isinstance(demo, dict) or not json_mode:
                state["demos"].append(demo)
            else:
                state["demos"].append(demo.toDict())

        state["signature"] = self.signature.dump_state()
        state["lm"] = self.lm.dump_state() if self.lm else None
        return state
    
    async def _lm_token_stream(self, lm, messages: list[dict[str, str]], **kwargs) -> AsyncIterator[str]:
        #TODO: extend to new lm class
        kwargs = dict(kwargs)
        #cache = kwargs.pop("cache", None)
        #num_retries  = int(kwargs.pop("num_retries", 3))
        if not lm.model or not lm.api_base:
            # fallback non-streaming
            outputs = await lm.acall(messages=messages, **kwargs)
            text = (outputs.get("text") if isinstance(outputs, dict) else str(outputs))
            if text:
                yield text
            return
        stream = litellm.acompletion(
        model=lm.model,
        messages=messages,
        stream=True,
        api_base=lm.api_base,
        api_key=getattr(lm, "api_key", None),
        **kwargs,
        )
        async for chunk in stream:
            piece = None
            try:
                piece = chunk.choices[0].delta.content
            except Exception:
                piece = None
            if piece:
                yield piece

    async def _lm_caller(self, lm, config, signature, messages, out_field_name):
        out_anno = signature.output_fields[out_field_name].annotation
        origin = get_origin(out_anno)
        should_be_list = (origin is list)
        print(should_be_list)
        lmtextpipe_otuput = []
        raw_tokens = []
        output_split = self.grandularity if should_be_list else WordOutput

        q: Queue[str | None] = Queue()
        pipe = LMTextPipe(queue=q, output_type=output_split)

        async def feeder():
            # feed raw tokens into the pipe
            async for tok in self._lm_token_stream(lm, messages, **config):
                if tok:
                    raw_tokens.append(tok)
                    await q.put(tok)
            await q.put(None)
        
        feeder_task = asyncio.create_task(feeder())
        try:
            async for part in pipe.process():
                seg = getattr(part, "content", "") or ""
                if seg:
                    lmtextpipe_otuput.append(seg)
        finally:
                await feeder_task

        final_text = "".join(raw_tokens).strip()
        return final_text, (lmtextpipe_otuput if should_be_list else None)

    async def _lm_stream_caller(self, lm, config, signature, messages, out_field_name):
        out_anno = signature.output_fields[out_field_name].annotation
        origin = get_origin(out_anno)
        should_be_list = (origin is list)
        
        raw_tokens = []
        output_split = self.grandularity if should_be_list else WordOutput

        q: Queue[str | None] = Queue()
        pipe = LMTextPipe(queue=q, output_type=output_split)

        async def feeder():
            try:
                async for tok in self._lm_token_stream(lm, messages, **config):
                    if tok:
                        raw_tokens.append(tok)
                        await q.put(tok)
            finally:
                await q.put(None)

        #streaming case
        
        async def gen() -> AsyncIterator[str]:
            feeder_task = asyncio.create_task(feeder())
            try:
                async for part in pipe.process():
                    seg = part.content
                    if seg:
                        yield seg
            finally:
                await feeder_task
        return gen()
    
    def _format_messages(self, signature, demos, inputs):
        #TODO: No support for few shot(Only can be access from the system intruction) 
        messages = []
        sys_msg = {"role": "system", "content": (signature.instructions or "").strip()}
        messages.append(sys_msg)
        blocks = []
        for name in signature.input_fields:
            value = str(inputs.get(name, "")).strip()
            if value is None:
                continue
            blocks.append(str(value).strip())
        user_text = "\n".join(block for block in blocks if block)
        messages.append({"role": "user", "content": user_text})
        output_messages = split_message_content_for_custom_types(messages)
        print(f"Formated messages: {output_messages}")
        return output_messages
    #TODO:Temp function. Remove it if found the better way
    def wrap_as_prediction(self, completions, signature) -> Prediction:
        pred = Prediction.from_completions(completions, signature=signature)
        (key, field) = next(iter(signature.output_fields.items()))
        if get_origin(field.annotation) is list:
            # Override the instance attribute so pred.answer is list[str]
            try:
                object.__setattr__(pred, key, pred._completions[key])
            except Exception:
                # If Prediction uses __slots__ / guarded __setattr__, skip (fallback: use pred._completions[key])
                pass
        return pred
    
    def _parse_messages(self, signature: type[Signature], completion: str, segments:None) -> dict[str, Any]:
        try:
            (key, field) = next(iter(signature.output_fields.items()))
            if segments is not None:
                items = [s.strip() for s in segments if s and s.strip()]
                return {key: items}
            else:
                return {key: [completion]}
        except Exception as e:
            raise ParseError(
                        signature=signature,
                        lm_response=completion,
                        message=f"Parse Error for field[{key}]: {e}",
                    )
#TODO: Solve circular import. This should be the expect pattern for granularity
# class SplitType: 
#     output_type = None
# class NewlineSplit(SplitType):
#     output_type = LineOutput
# class NewSentenceSplit(SplitType):
#     output_type = SentenceOutput 
# class NewWordSplit(SplitType): 
#     output_type = WordOutput    
# class NewFullTextSplit(SplitType): 
#     output_type = FullTextOutput 

def serialize_object(obj):
    """
    Recursively serialize a given object into a JSON-compatible format.
    Supports Pydantic models, lists, dicts, and primitive types.
    """
    if isinstance(obj, BaseModel):
        # Use model_dump with mode="json" to ensure all fields (including HttpUrl, datetime, etc.)
        # are converted to JSON-serializable types (strings)
        return obj.model_dump(mode="json")
    elif isinstance(obj, list):
        return [serialize_object(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(serialize_object(item) for item in obj)
    elif isinstance(obj, dict):
        return {key: serialize_object(value) for key, value in obj.items()}
    else:
        return obj
