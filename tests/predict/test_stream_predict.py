import asyncio
import copy
import enum
import time
import types
from datetime import datetime
from unittest.mock import patch

import orjson
import pydantic
import pytest
from litellm import ModelResponse
from pydantic import BaseModel, HttpUrl

import dspy
from dspy import Predict, Signature, StreamPredict
from dspy.predict.predict import serialize_object
from dspy.utils.dummies import DummyLM

def test_initialization_with_string_signature():
    signature_string = "input1, input2 -> output"
    predict = Predict(signature_string)
    expected_instruction = "Given the fields `input1`, `input2`, produce the fields `output`."
    assert predict.signature.instructions == expected_instruction
    assert predict.signature.instructions == Signature(signature_string).instructions

def test_call_method():
    predict_instance = Predict("input -> output")
    lm = DummyLM([{"output": "test output"}])
    dspy.settings.configure(lm=lm)
    result = predict_instance(input="test input")
    assert result.output == "test output"

def test_forward_method():
    program = Predict("question -> answer")
    dspy.settings.configure(lm=DummyLM([{"answer": "No more responses"}]))
    result = program(question="What is 1+1?").answer
    assert result == "No more responses"


def test_forward_method2():
    program = Predict("question -> answer1, answer2")
    dspy.settings.configure(lm=DummyLM([{"answer1": "my first answer", "answer2": "my second answer"}]))
    result = program(question="What is 1+1?")
    assert result.answer1 == "my first answer"
    assert result.answer2 == "my second answer"

def test_default_path_still_uses_predict_and_adapter_is_compatible():
    with dspy.context(adapter=dspy.ChatAdapter(), lm=DummyLM([{"answer": "No more responses"}])):
        sp = StreamPredict("question -> answer", enable_list_streaming=False, enable_text_streaming=False)
        out = sp(question="What is 1+1?")
        assert isinstance(out, dspy.Prediction)
        assert out.answer == "No more responses"

@pytest.mark.asyncio
async def test_lm_usage_with_async():
    program = Predict("question -> answer")

    original_aforward = program.aforward

    async def patched_aforward(self, **kwargs):
        await asyncio.sleep(1)
        return await original_aforward(**kwargs)

    program.aforward = types.MethodType(patched_aforward, program)

    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), track_usage=True):
        with patch(
            "litellm.acompletion",
            return_value=ModelResponse(
                choices=[{"message": {"content": "[[ ## answer ## ]]\nParis"}}],
                usage={"total_tokens": 10},
            ),
        ):
            coroutines = [
                program.acall(question="What is the capital of France?"),
                program.acall(question="What is the capital of France?"),
                program.acall(question="What is the capital of France?"),
                program.acall(question="What is the capital of France?"),
            ]
            results = await asyncio.gather(*coroutines)
            assert results[0].answer == "Paris"
            assert results[1].answer == "Paris"
            assert results[0].get_lm_usage()["openai/gpt-4o-mini"]["total_tokens"] == 10
            assert results[1].get_lm_usage()["openai/gpt-4o-mini"]["total_tokens"] == 10
            assert results[2].get_lm_usage()["openai/gpt-4o-mini"]["total_tokens"] == 10
            assert results[3].get_lm_usage()["openai/gpt-4o-mini"]["total_tokens"] == 10
