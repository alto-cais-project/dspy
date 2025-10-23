from abc import ABC, abstractmethod
from enum import Enum
import asyncio
from typing import Any, TypeVar, Generic, Self,List
from alto.utils import BytesSerializable
from alto.ancestry import AncestryTag
from alto import ancestry_pb2
from alto.lmtextpipe import (
    LMPartialOutput,
    FullTextOutput,
    SentenceOutput,
    LineOutput,
    WordOutput
)
import logging
class QueueItemType(Enum):
    CUSTOM_DATA = 1
    PARTIAL_TEXT = 2
    RESOLVED_TEXT = 3
    END_OF_PARTIAL_OUTPUT = 4
    END_OF_STREAM = 5
    JOINED = 6
    STREAM = 7
    END = 8
    EMPTY = 9
    LM_INPUT = 10
    CTRL_MESSAGE = 11
    LOG_MSG = 12
    NO_OP = 13
    LIST = 14


class CtrlMessageType(Enum):
    RECEIVED_MSG = 1
    SENT_MSG = 2
    FINISHED_PROCESSING = 3
    STARTING_PROCESSING = 4
    LM_MSG = 5


class QueueObject(ABC):
    @abstractmethod
    def get_inner(self) -> Any:
        pass

    @abstractmethod
    def get_type(Self) -> QueueItemType:
        pass

    def __eq__(self, value: object) -> bool:
        return False

    # TODO: only stream queue object type should be overriding this
    def __aiter__(self):
        return self

    def __anext__(self):
        raise NotImplementedError
QueueObjectA = TypeVar("QueueObjectA", bound=QueueObject)
QueueObjectB = TypeVar("QueueObjectB", bound=QueueObject)    
AsyncQueueItemT = TypeVar('AsyncQueueItemT')

class ObjectList(Generic[QueueObjectA], QueueObject):
    def __init__(self, items: List[QueueObjectA]):
        self._inner = items
    
    def to_bytes(self: Self) -> bytes:
        return NotImplemented

    @classmethod
    def from_bytes(cls: type[Self], buf: bytes) -> Self:
        return NotImplemented

    def get_inner(self) -> Any:
        return self._inner

    def get_type(self) -> QueueItemType:
        return QueueItemType.LIST

    def __eq__(self, value: object) -> bool:
        if isinstance(value, ObjectList):
            return self._inner == value.get_inner()
        return False
    
    def __len__(self) -> int:
        return len(self._inner)
    
    def __getitem__(self, index) -> QueueObject:
        return self._inner[index]
    
    def __iter__(self):
        return iter(self._inner)
    
    def __str__(self) -> str:
        return f"QueueObjectList[{len(self._inner)} items,{str(self._inner)}]"
    
class LMInputObject(QueueObject):
    def __init__(self,
                 prompt_key: str,
                 format_args: list[str] = [],
                 format_kwargs: dict[str, str] = {},
                 output_queues: list[asyncio.Queue] = []):
        self._prompt_key = prompt_key
        self._format_args = format_args
        self._format_kwargs = format_kwargs
        self._output_queues = output_queues

    @property
    def prompt_key(self):
        return self._prompt_key

    @property
    def format_args(self):
        return self._format_args

    @property
    def format_kwargs(self):
        return self._format_kwargs

    @property
    def output_queues(self):
        return self._output_queues

    def __str__(self):
        return f"LMInput[{self._prompt_key}, {self._format_args}, {self._format_kwargs}]"

    def get_inner(self) -> Any:
        return (self._prompt_key,
                self._format_args,
                self._format_kwargs,
                self._output_queues)

    def get_type(self) -> QueueItemType:
        return QueueItemType.LM_INPUT

    def __eq__(self, value: object) -> bool:
        if isinstance(value, LMInputObject):
            return self._format_args == value.format_args \
                and self._format_kwargs == value.format_kwargs
        return False
    
SerializableObjectA = TypeVar("SerializableObjectA", bound=BytesSerializable)


class CustomObject(Generic[SerializableObjectA], QueueObject,
                   BytesSerializable):
    def __init__(self, inner: SerializableObjectA):
        self._inner = inner

    def __str__(self):
        return f"Custom[{str(self._inner)}]"

    def get_inner(self) -> Any:
        return self._inner

    def get_type(self) -> QueueItemType:
        return QueueItemType.CUSTOM_DATA

    def __eq__(self, value: object,) -> bool:
        if isinstance(value, CustomObject):
            return self._inner == value.get_inner()
        return False
    
class PartialTextItem(QueueObject):
    def __init__(self, txt: str):
        self._txt = txt

    def __str__(self):
        return f"Partial[{self._txt}]"

    def get_inner(self) -> Any:
        return self._txt

    def get_type(self) -> QueueItemType:
        return QueueItemType.PARTIAL_TEXT
    
class QueueItem(QueueObject):
    _item: QueueObject
    _ancestry: AncestryTag

    def __init__(self, item: QueueObject, ancestry: AncestryTag):
        self._item = item
        self._ancestry = ancestry

    @property
    def item(self):
        return self._item

    @item.setter
    def item(self, new_item):
        self._item = new_item

    @property
    def ancestry(self):
        return self._ancestry

    @ancestry.setter
    def ancestry(self, ancestry: AncestryTag):
        self._ancestry = ancestry

    def __str__(self):
        return f"QueueItem[{str(self._item)},{str(self._ancestry)}]"

    def __eq__(self, value: object) -> bool:
        if isinstance(value, QueueItem):
            ret = self._item == value.item and self._ancestry == value.ancestry
            return ret
        return False

    def get_inner(self) -> Any:
        return self._item.get_inner()

    def get_type(self) -> QueueItemType:
        return self._item.get_type()
    
class EmptyBytes(QueueObject):
    def __init__(self, typ: QueueItemType):
        assert (
            typ == QueueItemType.END_OF_PARTIAL_OUTPUT or
            typ == QueueItemType.END_OF_STREAM or
            typ == QueueItemType.END or
            typ == QueueItemType.EMPTY or
            typ == QueueItemType.NO_OP
        )
        self._typ = typ

    def __str__(self):
        return f"EmptyBytes[{self._typ}]"

    def get_inner(self) -> Any:
        return None

    def get_type(self) -> QueueItemType:
        return self._typ


class EndOfPartialEmpty(EmptyBytes):
    def __init__(self):
        super().__init__(QueueItemType.END_OF_PARTIAL_OUTPUT)


class NoOpEmpty(EmptyBytes):
    def __init__(self):
        super().__init__(QueueItemType.NO_OP)


class EndofStreamEmpty(EmptyBytes, BytesSerializable):
    def __init__(self):
        super().__init__(QueueItemType.END_OF_STREAM)

    def __eq__(self, value: object) -> bool:
        if isinstance(value, EndofStreamEmpty):
            return True
        return False

    def get_inner(self) -> Any:
        return EndofStreamEmpty()  # return new version that can be serialized

    def to_bytes(self: Self) -> bytes:
        text_segment_proto = ancestry_pb2.TextSegment()
        text_segment_proto.text = ""
        text_segment_proto.text_type = ancestry_pb2.TextType.EMPTY
        return text_segment_proto.SerializeToString()

    @classmethod
    def from_bytes(cls: type[Self], buf: bytes) -> Self:
        text_segment_proto = ancestry_pb2.TextSegment()
        text_segment_proto.ParseFromString(buf)
        if text_segment_proto.text_type != ancestry_pb2.TextType.EMPTY:
            logging.info(f"Invalid data to deserialize into "
                         f"EndOfStreamEmpty: {text_segment_proto.text_type}")
            raise ValueError("Invalid data to deserialize into EndOfStreamEmpty")
        return EndofStreamEmpty()


class EndEmpty(EmptyBytes):
    def __init__(self):
        super().__init__(QueueItemType.END)

    def __eq__(self, value: object) -> bool:
        if isinstance(value, EndEmpty):
            return True
        return False


class EmptyObject(EmptyBytes):
    def __init__(self):
        super().__init__(QueueItemType.EMPTY)

    def __str__(self):
        return "EmptyObject"

    def __eq__(self, value: object) -> bool:
        if isinstance(value, EmptyObject):
            return True
        return False
    
class ResolvedText(QueueObject, BytesSerializable):
    def __init__(self, txt: LMPartialOutput):
        self._txt = txt

    def __str__(self):
        return f"Resolved[{self._txt}]"

    def get_inner(self) -> Any:
        return self._txt

    def get_type(self) -> QueueItemType:
        return QueueItemType.RESOLVED_TEXT

    def __eq__(self, value: object) -> bool:
        if isinstance(value, ResolvedText):
            return self._txt == value.get_inner()
        return False

    def to_bytes(self: Self) -> bytes:
        # turn to proto
        text_segment_proto = ancestry_pb2.TextSegment()
        if self._txt.content == "":
            logging.warning("RESOLVED TEXT EMPTY")
        text_segment_proto.text = self._txt.content
        if self._txt.delimeter_name == "FullText":
            text_segment_proto.text_type = ancestry_pb2.TextType.FULL_TEXT
        elif self._txt.delimeter_name == "Sentence":
            text_segment_proto.text_type = ancestry_pb2.TextType.SENTENCE
        elif self._txt.delimeter_name == "Line":
            text_segment_proto.text_type = ancestry_pb2.TextType.LINE
        elif self._txt.delimeter_name == "Word":
            text_segment_proto.text_type = ancestry_pb2.TextType.WORD
        else:
            logging.error(f"Unknown text type: {self._txt.delimeter_name}")
            text_segment_proto.text_type = ancestry_pb2.TextType.CUSTOM
        return text_segment_proto.SerializeToString()

    @classmethod
    def from_bytes(cls: type[Self], buf: bytes) -> Self:
        # deserialize from proto
        try:
            text_segment_proto = ancestry_pb2.TextSegment()
            text_segment_proto.ParseFromString(buf)
            if text_segment_proto.text_type == ancestry_pb2.TextType.FULL_TEXT:
                lm_partial_output = FullTextOutput(text_segment_proto.text)
            elif text_segment_proto.text_type == ancestry_pb2.TextType.SENTENCE:
                lm_partial_output = SentenceOutput(text_segment_proto.text)
            elif text_segment_proto.text_type == ancestry_pb2.TextType.LINE:
                lm_partial_output = LineOutput(text_segment_proto.text)
            elif text_segment_proto.text_type == ancestry_pb2.TextType.WORD:
                lm_partial_output = WordOutput(text_segment_proto.text)
            elif text_segment_proto.text_type == ancestry_pb2.TextType.CUSTOM:
                lm_partial_output = LMPartialOutput(text_segment_proto.text)
            else:
                raise ValueError(f"Unknown text type: {text_segment_proto.text_type}")
            return cls(lm_partial_output)
        except:
            raise ValueError("Invalid data to deserialize into TextSegment")
        
class CollectedItem(object):
    """Colllected item for debugging purposes.

    Attributes:
        _putter: The name of the node that put the item.
        _item: The item that was put, of type QueueItem.

    """
    def __init__(self, putter: str, item: QueueItem):
        self._putter = putter
        self._item = item

    def __str__(self):
        return f"CollectedItem[{self._putter}: {str(self._item)}]"

    @property
    def putter(self):
        """The name of the node that put the item."""
        return self._putter

    @property
    def item(self):
        """The item that was put, of type QueueItem."""
        return self._item