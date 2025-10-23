from typing import Generic, Any, Self, List, final, AsyncIterator
import asyncio
import os
import logging
import json
import time
from alto.base import QueueItemType, QueueObject, CtrlMessageType,QueueItem,ResolvedText, QueueObjectA, QueueObjectB, AsyncQueueItemT
from alto.ancestry import AncestryTag
from alto.utils import BytesSerializable
from alto.nodes import AltoRequestProcessingEvent
from collections import namedtuple

SIZE_BYTES = 8
DEFAULT_OUTGOING_BATCH_SIZE = 1
ZERO = 0

WriteLogEvent = namedtuple("WriteLogEvent", ("start", "end", "id", "name", "tag"))

def default_serialization_func(
        obj: BytesSerializable
        ) -> bytes:
    return obj.to_bytes()

class AsyncInputQueue:
    def __init__(self, 
                 socket: str, 
                 reader, 
                 writer,
                 data_class: type[BytesSerializable],
                 wrap_in_custom: bool,
                 logger=logging):
        self._socket = socket
        self._wrap_in_custom = wrap_in_custom
        self._queue_name = (
            None  # Should be set after socket connection has been established
        )
        self._reader = reader
        self._writer = writer
        self._data_class = data_class
        self._partial_data = b""
        self.logger = logger

    @property
    def wrap_in_custom(self):
        return self._wrap_in_custom

    @property
    def socket(self):
        return self._socket

    @property
    def queue_name(self):
        return self._queue_name

    @queue_name.setter
    def queue_name(self, name):
        self._queue_name = name

    @property
    def data_class(self):
        return self._data_class

    @data_class.setter
    def data_class(self, data_class):
        self._data_class = data_class

    async def read(self, size: int):
        data = await self._reader.read(size)
        #logging.debug(f"data in read({size}) = {data}")
        #if len(data) == 0:
            #print(f"Received close on {self._queue_name}")
        return data

    async def close(self):
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
            self._writer = None

    def is_closed(self):
        return self._writer is None

    def __del__(self):
        pass


class AsyncOutputQueue:
    def __init__(
        self,
        socket: str,
        stage_id: int,
        replica_id: int,
        output_dir: str,
        logger=logging,
    ):
        self._socket = socket
        self._queue_name = (
            None  # Should be set after socket connection has been established
        )
        self._stage_id = stage_id
        self._replica_id = replica_id
        self._end_times = {}
        self._writer = None
        self._active_writes = {}
        self._condition = asyncio.Condition()
        self._log_event_starts = {}
        self._write_log = []
        self._output_dir = output_dir
        self._write_lock = asyncio.Lock()
        self.logger = logger
        self._default_serialization_func = default_serialization_func

    def set_serialization_func(self, serialization_func):
        self._default_serialization_func = serialization_func

    @property
    def socket(self):
        return self._socket

    @property
    def queue_name(self):
        return self._queue_name

    @queue_name.setter
    def queue_name(self, name):
        self._queue_name = name

    async def wait_for_writes_to_drain(self):
        async with self._condition:
            await self._condition.wait_for(lambda: len(self._active_writes) == 0)

    async def record_write_start(self, 
                                 name, 
                                 identifier: AncestryTag,
                                 tag=None):
        """
        Indicates something will be written to this queue in the future.

        Write is on behalf of the input identifier; there may be multiple
        outgoing writes for the same input identifier (e.g., when an operation
        adds a level of ancestry).
        It is up to the caller to ensure that record_write_start is only called
        with unique identifiers, and that record_write_complete is called at the
        right time.

        Args:
            name: Name of the write operation
            identifier: AncestryTag of the input object
            tag: Optional tag to distinguish different events with the same
                identifier
        """
        async with self._condition:
            if identifier in self._active_writes:
                raise ValueError(f"Already have an active write for {identifier}")
            self._active_writes[identifier] = 1
            if tag:
                event_key = (identifier, name, tag)
                if event_key in self._log_event_starts:
                    raise ValueError(f"Already have a start time for {event_key}")
                self._log_event_starts[event_key] = time.monotonic_ns() / 1e3

    async def record_write_complete(self, 
                                    name, 
                                    identifier: AncestryTag, 
                                    tag=None):
        """
        Indicates that a write operation has completed.

        The write operation is on behalf of the identifier;
        there may actually be multiple writes for the same identifier.
        It is up to the caller to ensure that record_write_start is called
        before any of writes, and that record_write_complete is called after
        all of the writes.

        Args:
            name: Name of the write operation
            identifier: AncestryTag of the input object
            tag: Optional tag to distinguish different events with the same
                identifier
        """
        async with self._condition:
            end_ts = time.monotonic_ns() / 1e3
            
            # record to processing log
            if tag:
                event_key = (identifier, name, tag)
                if event_key not in self._log_event_starts:
                    raise ValueError(f"No start time for {event_key}")
                start_ts = self._log_event_starts[event_key]
                del self._log_event_starts[event_key]
                log_event = WriteLogEvent(start_ts, end_ts, identifier, name, tag)
                self._write_log.append(log_event)

            # decrement active write for queues
            if identifier in self._active_writes:
                self.logger.debug(
                    f"Queue {name} decrementing "
                    f"self._active_writes[{identifier}] from "
                    f"{self._active_writes[identifier]} -> "
                    f"{self._active_writes[identifier] - 1}"
                )
                self._active_writes[identifier] -= 1
                if self._active_writes[identifier] == 0:
                    del self._active_writes[identifier]
                    self.logger.debug(
                        f"Queue {name} self._active_writes[{identifier}] "
                        f"reached 0, signaling condition variable"
                    )
                    self._condition.notify_all()
            else:
                raise ValueError(f"No active write for {identifier}")

    async def connect(self):
        _, writer = await asyncio.open_unix_connection(self._socket)
        self._writer = writer

    async def write(self, 
                    data: list[bytes],
                    ):
        if self._writer is not None:
            async with self._write_lock:
                # lock here ensures that writes are not interleaved
                for b in data:
                    self._writer.write(b)
                    await self._writer.drain()
        else:
            raise ValueError("No writer to write to!")

    async def write_finished_processing_object(self,
                                               metadata: AltoRequestProcessingEvent) -> int:
        """
        Writes just the metadata object, with info about completions, onto the
        queue.

        """
        batch_size = DEFAULT_OUTGOING_BATCH_SIZE.to_bytes(SIZE_BYTES,
                                                          byteorder="little")
        metadata_bytes = metadata.alto_metadata_bytes()
        metadata_size = len(metadata_bytes).to_bytes(SIZE_BYTES,
                                                     byteorder="little")
        data_size = ZERO.to_bytes(SIZE_BYTES, byteorder="little")
        try:
            await self.write([batch_size, metadata_size, metadata_bytes,
                             data_size])
            return 0
        except ValueError:
            self.logger.warning(
                f"Failed to write object to socket with name: {self._socket}"
            )
            return -1

    async def write_object(self,
                           serializable_object: BytesSerializable,
                           ancestry_tag: AncestryTag,
                           ) -> int:
        """
        Writes a serializable object to the queue.

        Prepends the object with metadata containing the ancestry tag.

        Args:
           serializable_object: Object to write to the queue 
           ancestry_tag: AncestryTag of the object
        """
        # if any other metadata needs to go, add it here

        batch_size = DEFAULT_OUTGOING_BATCH_SIZE.to_bytes(SIZE_BYTES, byteorder="little")
        metadata_bytes = ancestry_tag.to_bytes()
        metadata_size = len(metadata_bytes).to_bytes(SIZE_BYTES,
                                                     byteorder="little")
        serialized_bytes = self._default_serialization_func(serializable_object)
        data_size = len(serialized_bytes).to_bytes(SIZE_BYTES, byteorder="little")
        logging.debug(f"Writing object to {self._socket} with tag "\
        f"{ancestry_tag},  data size {len(serialized_bytes)}")
        if len(serialized_bytes) == 0:
            logging.debug(f"Writing empty object to {self._socket} with tag "\
            f"{ancestry_tag}")
            data_size_read = int.from_bytes(data_size,
                                            byteorder="little")
            logging.debug(f"Data size read: {data_size_read}")
        try:
            await self.write([batch_size, metadata_size, metadata_bytes, data_size, serialized_bytes])
            return 0
        except ValueError:
            self.logger.warning(
                f"Failed to write object to socket with name: {self._socket}"
            )
            return -1

    async def shutdown(self):
        if self._writer:
            #print(f"Writing EOF for queue {self.socket}...")
            self._writer.write_eof()
            await self._writer.drain()
            #print(f"Finished writing EOF for queue {self.socket}")

            self.logger.debug(f"Closing queue {self.socket}...")
            self._writer.close()
            await self._writer.wait_closed()
            self._writer = None
            #print(f"Finished closing queue {self.socket}")
        else:
            self.logger.debug(
                f"Can't close {self.socket} because self._writer is None!")

    async def close(self):
        if self._writer:
            #print(f"Closing queue {self.socket}...")
            self._writer.close()
            await self._writer.wait_closed()
            #print(
            #    f"Queue {self.socket} has been closed and self._writer set to None"
            #)
            self._writer = None

        if len(self._write_log) == 0 or not self._output_dir:
            return

        def to_json(event):
            return {
                "name": event.name,
                "ph": "X",
                "ts": event.start,
                "dur": event.end - event.start,
                "pid": self._stage_id,
                "tid": self._replica_id + 1,
            }

        log = []
        for event in self._write_log:
            log.append(to_json(event))

        timing_file = os.path.join(
            self._output_dir,
            ".".join(
                [
                    f"queue={self._queue_name}",
                    f"stage={self._stage_id}",
                    f"replica={self._replica_id}",
                    "json",
                ]
            ),
        )
        logging.info(f"Writing {len(log)} events to {timing_file}...")
        with open(timing_file, "w") as f:
            json.dump(log, f, indent=4)
        logging.info(f"Finished writing log to {timing_file}")


class AsyncQueueListener:
    def __init__(self, 
                 socket: str,
                 data_class: type[BytesSerializable]):
        self._socket = socket
        self._server_task = None
        self._queue = asyncio.Queue()
        self._data_class = data_class
        # check if data_class is ResolvedText
        if data_class == ResolvedText:
            #print(f"For socket {socket}, setting wrap_in_custom as False for data_class {data_class}")
            self._wrap_in_custom = False
        else:
            #print(f"For socket {socket}, setting wrap_in_custom as True for data_class {data_class}")
            self._wrap_in_custom = True

    @property
    def wrap_in_custom(self):
        return self._wrap_in_custom

    async def handle_client(self, reader, writer):
        logging.debug(f"AsyncQueueListener {self._socket} received new client")
        await self._queue.put((reader, writer))

    async def start_server(self):
        server = await asyncio.start_unix_server(self.handle_client, path=self._socket)

        logging.debug(f"Starting AsyncQueueListener {self._socket} server...")
        self._server_task = asyncio.create_task(self.run_server(server))

    async def run_server(self, server):
        logging.debug(f"AsyncQueueListener {self._socket} server running")
        try:
            async with server:
                await server.serve_forever()
        except asyncio.CancelledError as e:
            return

    async def accept(self) -> AsyncInputQueue:
        logging.debug(
            f"AsyncQueueListener {self._socket} waiting for client connection..."
        )
        reader, writer = await self._queue.get()
        logging.debug(f"AsyncQueueListener {self._socket} received client connection")
        return AsyncInputQueue(self._socket, reader, writer, self._data_class, self._wrap_in_custom)

    async def close(self):
        if self._server_task is None:
            logging.warning(f"Trying to close server before it has been started")
        else:
            self._server_task.cancel()
            await self._server_task
            self._server_task = None
        try:
            os.unlink(self._socket)
        except FileNotFoundError:
            pass  # Socket might not exist
        except OSError as e:
            logging.warning(f'Error removing socket: {e}')

    def __del__(self):
        if self._server_task is not None:
            logging.warning("AsyncQueueListener server has not been closed!")


class CtrlMessage(QueueObject):
    def __init__(self, ctrl_msg: CtrlMessageType, lm_prompt:str = "", lm_response:str = ""):
        self._msg = ctrl_msg
        self._lm_prompt = lm_prompt
        self._lm_response = lm_response
        self._timestamp = time.time()

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, ts: float):
        self._timestamp = ts

    def __str__(self):
        return f"CtrlMessage[{self._msg}], LMPrompt[{self._lm_prompt}], LMResponse[{self._lm_response}]"

    def get_inner(self) -> Any:
        return (self._msg, self._timestamp, self._lm_prompt, self._lm_response)

    def get_type(self) -> QueueItemType:
        return QueueItemType.CTRL_MESSAGE

    def __eq__(self, value: object) -> bool:
        if isinstance(value, CtrlMessage):
            return self._msg == value._msg
        return False

class Joined(Generic[QueueObjectA, QueueObjectB], QueueObject,
             BytesSerializable):
    # TODO: doesn't make sense that joined needs to implement these methods
    def to_bytes(self: Self) -> bytes:
        return NotImplemented

    @classmethod
    def from_bytes(cls: type[Self], buf: bytes) -> Self:
        return NotImplemented

    def __init__(self, a: QueueObjectA, b: QueueObjectB):
        self._inner = (a, b)

    def __str__(self):
        return f"Joined{(str(self._inner[0]), str(self._inner[1]))}"

    def get_inner(self) -> Any:
        return self._inner

    def get_type(self) -> QueueItemType:
        return QueueItemType.JOINED

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Joined):
            return self._inner == value.get_inner()
        return False



@final
class AsyncQueueFullException(Exception):
    pass


@final
class AsyncQueueEmptyException(Exception):
    pass


@final
class AsyncQueue(Generic[AsyncQueueItemT]):
    max_size: int

    _queue: asyncio.Queue[AsyncQueueItemT]

    def __init__(
        self: Self,
        max_size: int = 0
    ) -> None:

        self.max_size = max_size
        self._queue = asyncio.Queue(self.max_size)

    def size(
        self: Self
    ) -> int:

        return self._queue.qsize()

    def remaining(
        self: Self
    ) -> int:

        return -1 if (self.max_size == 0) else (self.max_size - self.size())

    def is_empty(
        self: Self
    ) -> bool:

        return self._queue.empty()

    def is_full(
        self: Self
    ) -> bool:

        return self._queue.full()

    def put_nowait(
        self: Self,
        item: AsyncQueueItemT
    ) -> None:

        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            raise AsyncQueueEmptyException()

    def get_nowait(
        self: Self
    ) -> AsyncQueueItemT:

        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            raise AsyncQueueEmptyException()

    async def put_async(
        self: Self,
        item: AsyncQueueItemT,
        timeout: float | None = None
    ) -> None:

        if (timeout is not None) and (timeout < 0):
            raise ValueError("Invalid input")

        try:
            await asyncio.wait_for(self._queue.put(item), timeout)
        except asyncio.TimeoutError:
            raise AsyncQueueFullException()

    async def get_async(
        self: Self,
        timeout: float | None = None
    ) -> AsyncQueueItemT:

        if (timeout is not None) and (timeout < 0):
            raise ValueError("Invalid input")

        try:
            return await asyncio.wait_for(self._queue.get(), timeout)
        except asyncio.TimeoutError:
            raise AsyncQueueEmptyException()

    def put_nowait_batch(
        self: Self,
        items: list[AsyncQueueItemT]
    ) -> None:

        if self.max_size > 0 and len(items) + self.size() > self.max_size:
            raise AsyncQueueFullException(
                f"Cannot add {len(items)} items to queue of size {self.size()} and maxsize {self.max_size}."
            )
        for item in items:
            self._queue.put_nowait(item)

    def get_nowait_batch(
        self: Self,
        num_items: int | None = None,
        max_num_items: int | None = None,
        all_items: bool = False
    ) -> list[AsyncQueueItemT]:

        if ((max_num_items is not None) + (num_items is not None) + all_items) != 1:
            raise ValueError("Invalid input")
        if (num_items is not None) and (num_items > self.size() or num_items < 0):
            raise AsyncQueueEmptyException(f"Cannot get {num_items} items from queue of size {self.size()}.")
        if max_num_items is not None:
            num_items = min(self.size(), max_num_items)
        if all_items:
            num_items = self.size()
        assert num_items is not None

        out = []
        try:
            for _ in range(num_items):
                out.append(self._queue.get_nowait())
        except AsyncQueueEmptyException:
            pass

        return out

    async def put_batch_async(
        self: Self,
        items: list[AsyncQueueItemT],
        timeout: float | None = None
    ) -> None:

        if (timeout is not None) and (timeout < 0):
            raise ValueError("Invalid input")
        if self.max_size > 0 and len(items) + self.size() > self.max_size:
            raise AsyncQueueFullException(
                f"Cannot add {len(items)} items to queue of size "
                f"{self.size()} and maxsize {self.max_size}."
            )

        for item in items:
            await self.put_async(item, timeout)

    async def get_batch_async(
        self: Self,
        num_items: int | None = None,
        max_num_items: int | None = None,
        all_items: bool = False,
        min_num_items: int | None = None,
        timeout: float | None = None
    ) -> list[AsyncQueueItemT]:

        if ((max_num_items is not None) + (num_items is not None) + all_items) != 1:
            raise ValueError("Invalid input")
        if (num_items is not None) and (num_items > self.size() or num_items < 0):
            raise AsyncQueueEmptyException(f"Cannot get {num_items} items from queue of size {self.size()}.")
        if (timeout is not None) and (timeout < 0):
            raise ValueError("Invalid input")
        if max_num_items is not None:
            num_items = min(self.size(), max_num_items)
            if min_num_items is not None:
                num_items = max(min_num_items, num_items)
        if all_items:
            num_items = self.size()
            if min_num_items is not None:
                num_items = max(min_num_items, num_items)
        assert num_items is not None

        out = []
        try:
            for _ in range(num_items):
                out.append(await asyncio.wait_for(self._queue.get(), timeout))
        except asyncio.TimeoutError:
            pass

        return out


class OutputQueue(AsyncOutputQueue):
    """
    Async queue that records items and yields QueueItem objects directly.
    """
    def __init__(self, name: str):
        super().__init__(name, stage_id=0, replica_id=0, output_dir="/local_ssd1/nmodugul")
        self.queue_name = name
        self.received_items: List[QueueItem] = []
        self.ancestry_groups = {}
        self._stream_q = asyncio.Queue()

    async def write_object(self, obj: Any, ancestry: AncestryTag):
        if ancestry is None:
            ancestry = AncestryTag()

        qitem = QueueItem(obj, ancestry)
        ancestry_key = str(ancestry)

        if ancestry_key not in self.ancestry_groups:
            self.ancestry_groups[ancestry_key] = []

        if isinstance(obj, ResolvedText):
            content = str(obj.get_inner())
        else:
            content = str(obj)

        self.ancestry_groups[ancestry_key].append({
            "type": type(obj).__name__,
            "content": content,
            "is_resolved": isinstance(obj, ResolvedText),
        })
        self.received_items.append(qitem)

        # push into the async stream queue for consumers
        await self._stream_q.put(qitem)

    async def stream(self) -> AsyncIterator[QueueItem]:
        """Async generator of QueueItem objects."""
        while True:
            qitem = await self._stream_q.get()
            if qitem is None:
                break
            yield qitem

    async def close(self):
        # Put None so .stream() breaks
        await self._stream_q.put(None)

class Stream(Generic[QueueObjectA], QueueObject, BytesSerializable):
    # TODO: doesn't make sense that stream needs to implement these methods
    def to_bytes(self: Self) -> bytes:
        return NotImplemented

    @classmethod
    def from_bytes(cls: type[Self], buf: bytes) -> Self:
        return NotImplemented

    def __init__(self, queue: AsyncQueue):
        self._queue = queue

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self._queue.get_async()
        if item.get_type() == QueueItemType.END_OF_STREAM:
            # print("Raising stop iteration of stream")
            raise StopAsyncIteration
        else:
            return item

    def get_inner(self) -> Any:
        return self._queue

    def get_type(self) -> QueueItemType:
        return QueueItemType.STREAM