import asyncio
import pydantic
from pydantic import BaseModel, ConfigDict 
import dspy
from dspy import StreamPredict
from ..alto.lmtextpipe import WordOutput, LineOutput, SentenceOutput, FullTextOutput
async def main():
    class SplitType: 
        output_type = None
    class NewlineSplit(SplitType):
        output_type = LineOutput
    class NewSentenceSplit(SplitType):
        output_type = SentenceOutput 
    class NewWordSplit(SplitType): 
        output_type = WordOutput    
    class NewFullTextSplit(SplitType): 
        output_type = FullTextOutput 
    lm = dspy.LM(
        "openai/open-orca/mistral-7b-openorca",
        api_base="http://localhost:8001/v1",
        api_key="fake-key",
        model_type='chat',
        temperature = 0.5,
        cache=False,
        )
    dspy.configure(lm=lm)
    # class QueryResult(pydantic.BaseModel):
    #     model_config = ConfigDict(arbitrary_types_allowed=True)
    #     granularity = "LineOutput"

    class TestSignature(dspy.Signature):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        input_text: str = dspy.InputField()
        output_text: list["LineOutput"] = dspy.OutputField()

    program = StreamPredict(TestSignature, is_streaming=True)

    output_stream1 = await program.aforward(input_text="is sun bright?")         
    async for chunk in output_stream1:
        print(f"stream output:{chunk} with type {type(chunk)}")

    program2 = StreamPredict(TestSignature, is_streaming=False)

    output_stream2 = await program2.aforward(input_text="is sun bright?")
    print(f"result: {output_stream2.output_text}")
    assert isinstance(output_stream2, dspy.Prediction)       
if __name__ == "__main__":
    asyncio.run(main())
