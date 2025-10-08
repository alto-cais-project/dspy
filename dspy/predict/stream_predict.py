import logging
import random

from pydantic import BaseModel

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.clients.base_lm import BaseLM
from dspy.clients.lm import LM
from dspy.dsp.utils.settings import settings
from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.signatures.signature import Signature, ensure_signature
from dspy.utils.callback import BaseCallback

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

    def __init__(self, signature: str | type[Signature], callbacks: list[BaseCallback] | None = None,enable_text_streaming: bool = False,
        enable_list_streaming: bool = False, **config):
        super().__init__(callbacks=callbacks)
        self.stage = random.randbytes(8).hex()
        self.signature = ensure_signature(signature)
        self.enable_text_streaming = enable_text_streaming
        self.enable_list_streaming = enable_list_streaming
        self._predict = Predict(signature, callbacks=callbacks, **config)
        self.config = config
    
    def forward(self, **kwargs):
        list_flag = self.enable_list_streaming
        text_flag = self.enable_text_streaming
        #TODO: Streaming both text and list
        if list_flag and text_flag:
            print(f"Option 1: list_flag = {list_flag} and text_flag = {text_flag}")
         #TODO: Only streaming list
        if list_flag and not text_flag:
            print(f"Option 2: list_flag = {list_flag} and text_flag = {text_flag}")

        return self._predict(**kwargs)


        

    async def aforward(self, **kwargs):
        list_flag = self.enable_list_streaming
        text_flag = self.enable_text_streaming

        #TODO: Streaming both text and list
        if list_flag and text_flag:
            print(f"Option 1: list_flag = {list_flag} and text_flag = {text_flag}")
         #TODO: Only streaming list
        if list_flag and not text_flag:
            print(f"Option 2: list_flag = {list_flag} and text_flag = {text_flag}")

        return self._predict(**kwargs)



