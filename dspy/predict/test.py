# simple_streampredict_demo.py
import asyncio
import dspy
from dspy import StreamPredict
from dspy.clients.base_lm import BaseLM
# --- YOUR StreamPredict class must be importable here ---
# from your_module import StreamPredict

# A tiny fake LM that satisfies your _lm_token_stream() fallback path:
# - No model/api_base -> it uses lm.acall(...) non-streaming mode
# - Returns a dict with "text", which your code joins/streams via LMTextPipe
class FakeLM(BaseLM):
    def __init__(self):
        # model="" ensures the fallback branch (no external server needed)
        super().__init__(model="")
        self.model = ""        # falsy, triggers non-streaming fallback in _lm_token_stream
        self.api_base = None   # falsy, same reason
        self.kwargs = {}

    async def acall(self, messages, **kwargs):
        # You can craft the text however you like; this is enough to demo
        # messages is a list of {"role": ..., "content": ...}, per your formatter
        return {"text": "Hello from FakeLM! 1 + 1 = 2."}

    def dump_state(self):
        return {"type": "FakeLM"}

# Define a simple signature
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: list[str]   = dspy.OutputField()

async def main():
    # Use FakeLM inside the DSPy context so your StreamPredict can find it
    with dspy.context(lm=FakeLM()):
        # --- Non-streaming usage: returns a dspy.Prediction ---
        non_streaming = StreamPredict(QA, is_streaming=False)
        pred = await non_streaming.aforward(question="What is 1+1?")
        print("[non-streaming] answer:", pred.answer)
        print(type(pred.answer))

        # --- Streaming usage: returns an async iterator of STRINGS (your current code) ---
        streaming = StreamPredict(QA, is_streaming=True)
        stream = await streaming.aforward(question="Stream the answer to 1+1, please.")
        chunks = []
        async for chunk in stream:
            # Your _lm_stream_caller currently yields raw string segments
            print("[stream chunk]", repr(chunk))
            chunks.append(chunk)
        print("[stream joined]", "".join(chunks))

if __name__ == "__main__":
    asyncio.run(main())

