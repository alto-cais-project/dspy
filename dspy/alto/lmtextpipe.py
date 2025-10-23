# -*- coding: utf-8 -*-
"""
Created on Wed July 31

@author: Deepti
"""
# how can we access the underlying engine in a thread safe way?
# e.g. appending to the version of LMTextPipe() at the full text level
# should append to all the buffers in some threadsafe way

import asyncio
from asyncio import Queue
import re
from .utils import BytesSerializable
from . import ancestry_pb2
from typing import AsyncIterator, Callable, TypeVar, Generic, ClassVar, final, Self
from collections.abc import Callable

def basic_strip_function(x: str):
    return x.strip()

class LMPartialOutput(object):
    _delimeter_tokens: ClassVar[[str]] = NotImplemented
    _delimeter_name: ClassVar[str] = NotImplemented
    _text_processor: ClassVar[Callable[[str], str] | None] = NotImplemented
    _content: str
    
    def __init__(self, content: str):
        if "" in self._delimeter_tokens:
            assert(len(self._delimeter_tokens) == 1)
        self._content = content
    
    def __eq__(
            self: Self,
            other,
            ) -> bool:
        return isinstance(other, LMPartialOutput) and \
                 self.delimeter_tokens == other._delimeter_tokens and \
                 self.delimeter_name == other._delimeter_name and \
                self.content == other.content
    @property
    def text_processor(self):
        return self._text_processor

    @property
    def delimeter_tokens(self):
        return self._delimeter_tokens

    @property
    def splitting_pattern(self):
        return '|'.join([re.escape(delim) for delim in self.delimeter_tokens])
    
    @property
    def delimeter_name(self):
        return self._delimeter_name

    @property
    def content(self):
        if self._text_processor is None:
            return self._content
        else:
            return self._text_processor(self._content)

    def __repr__(self):
        return f"LMPartialOutput.{self._delimeter_name}[{self.content!r}]"
@final
class FullTextOutput(LMPartialOutput, BytesSerializable):
    _delimeter_tokens = [""]
    _delimeter_name = "FullText"
    _text_processor = lambda self, x: x.strip()

    def to_bytes(self) -> bytes:
        text_segment_proto = ancestry_pb2.TextSegment()
        text_segment_proto.text = self.content
        text_segment_proto.text_type = ancestry_pb2.TextType.FULL_TEXT
        return text_segment_proto.SerializeToString()

    @classmethod
    def from_bytes(cls, buf: bytes):
        text_segment_proto = ancestry_pb2.TextSegment()
        text_segment_proto.ParseFromString(buf)
        if text_segment_proto.text_type != ancestry_pb2.TextType.FULL_TEXT:
            raise ValueError("TextSegment proto is not of type FULL_TEXT")
        return cls(text_segment_proto.text)
 
@final
class SentenceOutput(LMPartialOutput, BytesSerializable):
    _delimeter_tokens = ". "
    _delimeter_name = "Sentence"
    _text_processor = lambda self, x: x.strip()
    
    def to_bytes(self) -> bytes:
        text_segment_proto = ancestry_pb2.TextSegment()
        text_segment_proto.text = self.content
        text_segment_proto.text_type = ancestry_pb2.TextType.SENTENCE
        return text_segment_proto.SerializeToString()

    @classmethod
    def from_bytes(cls, buf: bytes):
        text_segment_proto = ancestry_pb2.TextSegment()
        text_segment_proto.ParseFromString(buf)
        if text_segment_proto.text_type != ancestry_pb2.TextType.SENTENCE:
            raise ValueError("TextSegment proto is not of type SENTENCE")
        return cls(text_segment_proto.text)

@final
class LineOutput(LMPartialOutput, BytesSerializable):
    """
    This class represents splitting on the same exact delimeters as Python's splitlines function (version 3.2 and higher).
    """
    _delimeter_tokens = ['\n', '\r\n', '\r', '\v', '\f', '\x85', '\u2028', '\u2029', '\x1c', '\x1d', '\x1e']
    _delimeter_name = "Line"
    _text_processor = lambda self, x: x.strip()

    def to_bytes(self) -> bytes:
        text_segment_proto = ancestry_pb2.TextSegment()
        text_segment_proto.text = self.content
        text_segment_proto.text_type = ancestry_pb2.TextType.LINE
        return text_segment_proto.SerializeToString()

    @classmethod
    def from_bytes(cls, buf: bytes):
        text_segment_proto = ancestry_pb2.TextSegment()
        text_segment_proto.ParseFromString(buf)
        if text_segment_proto.text_type != ancestry_pb2.TextType.LINE:
            raise ValueError("TextSegment proto is not of type LINE")
        return cls(text_segment_proto.text)

@final
class WordOutput(LMPartialOutput, BytesSerializable):
    _delimeter_tokens = [" "]
    _delimeter_name = "Word"
    _text_processor = lambda self, x: x.strip()

    def to_bytes(self) -> bytes:
        text_segment_proto = ancestry_pb2.TextSegment()
        text_segment_proto.text = self.content
        text_segment_proto.text_type = ancestry_pb2.TextType.WORD
        return text_segment_proto.SerializeToString()
    
    @classmethod
    def from_bytes(cls, buf: bytes):
        text_segment_proto = ancestry_pb2.TextSegment()
        text_segment_proto.ParseFromString(buf)
        if text_segment_proto.text_type != ancestry_pb2.TextType.WORD:
            raise ValueError("TextSegment proto is not of type WORD")
        return cls(text_segment_proto.text)

PartialOutputType = TypeVar('PartialOutputType', bound = "LMPartialOutput")
PartialOutputTypeB = TypeVar('PartialOutputTypeB', bound = "LMPartialOutput")

class LMTextPipe(Generic[PartialOutputType]):

    def __init__(self, 
            queue: Queue,
            output_type: type(PartialOutputType)
            ):
        self._queue = queue
        self._children = []
        self._emitted = []
        self._output_type = output_type
        self._output_type_instance = output_type("")
        self._buffered = ""

    def reset(self):
        self._emitted = []
        self._buffered = ""
        for child in self._children:
            child.reset()
    @property
    def buffered(self):
        return self._buffered
    
    @buffered.setter
    def buffered(self, new):
        self._buffered = new
    
    @property
    def delimeter_tokens(self):
        return self._output_type_instance.delimeter_tokens

    @property
    def splitting_pattern(self):
        return self._output_type_instance.splitting_pattern

    @property
    def delimeter_name(self):
        return self._output_type_instance.delimeter_name

    def split(
            self: Self,
            split_type: type[PartialOutputTypeB],
            ) -> 'LMTextPipe[PartialOutputTypeB]':
        assert(len(self._emitted) == 0)
        new_queue = Queue()
        new_text_pipe = LMTextPipe[PartialOutputTypeB](
                new_queue,
                split_type)
        self._children.append(new_text_pipe)
        return new_text_pipe

    @property
    def queue(self):
        return self._queue

    async def push_to_children(self, item: str | None):
        for child in self._children:
            await child.queue.put(item)

    async def process_buffered(self, process_last: bool) -> AsyncIterator[PartialOutputType]:
        """
        This method processes the current buffered string by:
        1. Splitting the string according to the delimeters,
        2. Adding each split to emitted,
        3. Yielding an item.
        4. Options are: 
            - process_last: Whether to emit the last split (after the delimeter) or save it in buffered.
            - Would be true in the case of having seen an '' character, indicating a full input in the parent.
            - Would be false in the case of having seen the delimeters, in which case the last segment may not have ended.
        Must NOT be called on "FullText" as that has a special case emitting time.
        """
        assert(self.delimeter_tokens != [""])
        if process_last:
            while self.buffered != "":
                res = re.split(self.splitting_pattern, self.buffered, maxsplit=1)
                assert(len(res) <= 2)
                part = res[0]
                if len(res) > 1:
                    self.buffered = res[1]
                else:
                    self.buffered = ""
                if part == "":
                    continue
                #print("\n****")
                #print(f"[{self.delimeter_name}] EMITTING: [{self._output_type(part)}]")
                #print("*****\n")
                self._emitted.append(part)
                yield self._output_type(part)
        else:
            while re.search(self.splitting_pattern, self.buffered):
                res = re.split(self.splitting_pattern, self.buffered, maxsplit=1)
                assert(len(res) == 2)
                part = res[0]
                self.buffered = res[1]
                if part == "":
                    continue
                #print("\n****")
                #print(f"[{self.delimeter_name}] EMITTING: [{self._output_type(part)}]")
                #print("*****\n")
                self._emitted.append(part)
                yield self._output_type(part)
    
    async def push_new_data_to_children(self, new_token: str):
        parts = re.split(self.splitting_pattern, new_token)
        for i, part in enumerate(parts):
            await self.push_to_children(part)
            if i != len(parts) - 1:
            # signals end of parent output to the child 
            # for last part, we don't know whether the parent output has ended
                await self.push_to_children('')
    
    async def process(self) -> AsyncIterator[PartialOutputType]:
        while True:
            next_token = await self._queue.get()
            if next_token is None:
                #print(f"[{self.delimeter_name}]: *****GOT NONE****")
                if self.buffered:
                    if self.delimeter_tokens == [""]:
                        if self.buffered != "":
                            self._emitted.append(self.buffered)
                            yield self._output_type(self.buffered)
                    else:
                        async for item in self.process_buffered(process_last = True):
                            yield item
                break
            elif next_token == '':
                if self.buffered:
                    if self.delimeter_tokens != [""]:
                        async for item in self.process_buffered(process_last=True):
                            yield item
                            continue
            else:
                self.buffered += next_token
                if self.delimeter_tokens != [""]:
                    search_res = re.search(self.splitting_pattern, self.buffered)
                else:
                    search_res = None
                #print(f"[{self.delimeter_name}]: new_token: [{next_token!r}], buffered: [{self.buffered!r}], search res: {search_res}, search res none?: {search_res is None} PROCESSING NEXT TOKEN")
                if self.delimeter_tokens == [""] or search_res is None:
                    #print(f"[{self.delimeter_name}]: new_token: [{next_token!r}], buffered: [{self.buffered!r}] PUSHING TO CHILD")
                    await self.push_to_children(next_token)
                else:
                    #print(f"[{self.delimeter_name}]: new_token: [{next_token!r}], buffered: [{self.buffered!r}] SPLITTING AND PUSHING")
                    await self.push_new_data_to_children(next_token)
                    async for item in self.process_buffered(process_last = False):
                        yield item
        #print("\n********\n")
        #print(f"[{self.delimeter_name}]: Pushing none to children")
        #print("\n********\n")
        await self.push_to_children(None)
        return
    
    async def add_to_queue(self, stream: AsyncIterator[str]):
        async for item in stream:
            await self.queue.put(item)
        await self.queue.put(None) 
                
# Example usage
async def main():
    async def lm_engine_example():
        prompt = "Hello world\nThis is an example\nOf a text stream\nextra text without newline"
        for char in prompt:
            #print(f"Yielding {char!r}")
            yield char
            await asyncio.sleep(0.01)  # Simulate async text generation
        next_part = "\nNewline test\nactual last part"
        yield next_part

    root_queue = Queue()
    root_node = LMTextPipe[FullTextOutput](root_queue, FullTextOutput)
    line_node = root_node.split(split_type = LineOutput)
    word_node = line_node.split(split_type = WordOutput)

    # Create tasks for processing each node
    async def process_node(node: LMTextPipe):
        async for item in node.process():
            print(f"{item}")

    root_process_task = asyncio.create_task(process_node(root_node))
    word_process_task = asyncio.create_task(process_node(line_node))
    line_process_task = asyncio.create_task(process_node(word_node))

    # Add the initial stream to the root node

    await root_node.add_to_queue(lm_engine_example())

    # Wait for both tasks to complete
    await asyncio.gather(root_process_task, line_process_task, word_process_task)
if __name__ == '__main__':
    asyncio.run(main())
