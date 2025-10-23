"""
Created on Wed August 15
@author: Deepti

This module contains classes for creating a dataflow tree for text processing.
Classes:
    CollectedItem: A collected item for debugging purposes.
    AsyncTreeNode: A node in the dataflow tree.
    Splitter: A node that splits text into parts based on a PartialOutputType.
    TextPipePmapEntry: A node that processes text-based queue items and modifies
    ancestry to add a level of ancestry corresponding to the count of partial
    text.
    DebugNode: A node that prints collects debugging items.

Functions:
    add_to_queue: Adds a stream to the input queue of a node.

"""
import asyncio
import logging
import time
import copy
import collections
from enum import Enum
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, AsyncIterator, Type,Self, Any
import re
from alto.ancestry import AncestryId, AncestryTag
from alto.base import  QueueItemType, PartialTextItem, QueueItem, EndOfPartialEmpty, EndofStreamEmpty, ResolvedText, EndEmpty, CollectedItem,QueueObject
from alto.lmtextpipe import  FullTextOutput, LMPartialOutput, LineOutput, WordOutput
from colorama import Fore, Style
from alto.ancestry_pb2 import FinishedProcessing

class TreeNodeKey(object):
    """A key for a node in the dataflow tree.
    Can optionally have a uid to denote different requests."""
    def __init__(self, key: str, uid: int = 0):
        self._key = key
        self._uid = uid
    def __hash__(self):
        return hash((self._key, self._uid))
    def __str__(self):
        return f"{self._key}_{self._uid}"
    @property
    def key(self):
        return self._key
    @property
    def uid(self):
        return self._uid
    @uid.setter
    def uid(self, val):
        self._uid = val

    def __eq__(self, other):
        if isinstance(other, TreeNodeKey):
            return self._key == other._key and self._uid == other._uid
        return False

class AsyncTreeNode(ABC):
    """A node in the dataflow tree.

    Attributes:
        _children: A list of references to children, of type AsyncTreeNode.
        _stop_event: An event that signals the node to stop processing.
        _queue: A queue of items to be processed.
        _collected_data: A list of collected items for debugging purposes.
        _debug: A boolean indicating whether debugging is turned on.

    """
    def __init__(self, key: TreeNodeKey, *args, **kwargs):
        self._key = key
        self._parent_queues: dict[TreeNodeKey, asyncio.Queue] = {}
        self._parent_poll_tasks = {}
        self._children: dict[TreeNodeKey, 'AsyncTreeNode'] = {}
        self._stop_event = asyncio.Event()
        self._collected_data: list[CollectedItem] = []
        self._debug = False
        self._collected_data_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._started = False
        # whether end events should be propagated to the process_queue_item
        # needed in the case of generate which attaches send nodes to
        # temporary subtrees
        self._propagate_end = False

    @property
    def key(self):
        """The key of the node."""
        return self._key

    @property
    def debug(self):
        """A boolean indicating whether debugging is turned on."""
        return self._debug

    @debug.setter
    def debug(self, val):
        """Set the debug value."""
        self._debug = val

    def print(self, string: str):
        """Prints a string if debugging is turned on."""
        if self.debug:
            my_str = f"{Fore.BLUE}Node[{str(self), str(self.key)}]: {Style.RESET_ALL}{string}"
            print(my_str)

    def add_parent_queue(self, parent_key: TreeNodeKey, queue: asyncio.Queue):
        """Adds a parent queue to the node."""
        # next time the node loops inside get_next, it will add
        # the task to poll from this queue
        self._parent_queues[parent_key] = queue

    async def remove_parent_queue(self, parent_key: TreeNodeKey):
        """Removes a parent queue from the node."""
        if parent_key in self._parent_poll_tasks:
            async with asyncio.TaskGroup() as tg:
                # TODO: does accessing parent poll tasks here require a lock?
                tg.create_task(self._parent_poll_tasks[parent_key].cancel())
                del self._parent_poll_tasks[parent_key]
        del self._parent_queues[parent_key]

    def add_child(self, child: 'AsyncTreeNode'):
        """Adds a child to the node."""
        self._children[child.key] = child
        child.add_parent_queue(self.key, asyncio.Queue())

    def get_children(self) -> dict[TreeNodeKey, 'AsyncTreeNode']:
        """Returns a list of references to children, mapped by name"""
        return self._children

    def collect_item(self, item: CollectedItem): 
        """Appends item to collected array."""
        self._collected_data.append(item)

    async def put(self, parent_key: TreeNodeKey, item: QueueItem):
        """Puts data onto input queue."""
        if parent_key not in self._parent_queues:
            logging.warning(f"[{self.key}] Parent key {parent_key} "\
                    f"not in parent queues")
            raise KeyError(f"[{self.key}] Parent key {parent_key}")
        await self._parent_queues[parent_key].put(item)
        # hopefully gives the child a chance to execute
        await asyncio.sleep(0)

    async def read_from_parent_queue(self, parent_key: TreeNodeKey) -> tuple[TreeNodeKey, QueueItem]:
        """Reads data from parent queue."""
        #self.print(f"Reading from {parent_key.key}")
        try:
            ret = await self._parent_queues[parent_key].get()
            #if ret.item.get_type() == QueueItemType.END:
                #self.print(f"Received END from {parent_key.key}")
        except Exception as e:
            self.print(f"WARNING when reading: {e}")

        return (parent_key, ret)

    async def get_next(self, collected_data: list[CollectedItem]) -> dict[TreeNodeKey,
                                     QueueItem]:
        """Gets next item from ANY parent queue. Like epoll"""

        # node should hve SOME parent queues when starting
        while not(self._stop_event.is_set()):
            received_end = False
            
            # add new tasks for all parent queues
            for parent_key in self._parent_queues:
                if parent_key not in self._parent_poll_tasks:
                    self._parent_poll_tasks[parent_key] = asyncio.create_task(self.read_from_parent_queue(parent_key))
            
            # get next events on queues
            if len(self._parent_poll_tasks) > 0:
                #parent_poll_keys = [str(parent) for parent in
                                    #self._parent_poll_tasks]
                #self.print(f"Waiting on {parent_poll_keys} parent items")
                finished, _ = await asyncio.wait(list(self._parent_poll_tasks.values()), return_when=asyncio.FIRST_COMPLETED)
                ret = {}
                
                for task in finished:
                    try:
                        parent_key, queue_item = task.result()
                    except Exception as e:
                        #import pdb; pdb.set_trace()
                        logging.warning(f"Task cancelled: {e}")
                    del self._parent_poll_tasks[parent_key]
                    # if the item is an end item, remove the parent queue
                    if queue_item.item.get_type() == QueueItemType.END:
                        #my_str = f"{Fore.RED}Node[{str(self), str(self.key)}]: {Style.RESET_ALL}RECEIVED END"
                        #print(my_str)
                        await self.remove_parent_queue(parent_key)

                        received_end = True
                        if self._propagate_end and len(self._parent_queues) > 0:
                            #self.print("PROPAGATING END")
                            # in cases where propagate_end is set,
                            # must inform process_queue_item of end event
                            # to propagate end to selective children
                            # not necessary on last end as handle_end will take
                            # care of it
                            ret[parent_key] = queue_item
                    else:
                        ret[parent_key] = queue_item

                if len(ret) > 0:
                    return ret
                
                # only handle end after receiving end from all parents
                # assumes handle end sets stop event
                if received_end and len(self._parent_queues) == 0:
                    #my_str = f"{Fore.RED}Node[{str(self), str(self.key)}]: {Style.RESET_ALL}RECEIVED END FROM ALL"
                    #print(my_str)
                    await self.set_stop_event()
                #else:
                    #keys = [(parent.key, parent.uid) for parent in
                    #        self._parent_queues]
                    #my_str = f"{Fore.BLUE}Node[{str(self), str(self.key)}]:{Style.RESET_ALL}Waiting on {keys}"
            # case where node has started but no parent queues have been set
            else:
                await asyncio.sleep(0.1)
        return {}

    async def set_stop_event(self):
        """Sets the stop event."""
        self._stop_event.set()

    @abstractmethod
    async def process_queue_item(self, 
        next_item: QueueItem,
                                 parent_key: TreeNodeKey,
                                 collected_data: list[CollectedItem]
                                 ) -> AsyncIterator[QueueItem]:
        """Processes the next item on the queue.
            
        Arguments:
            next_item: The next item on the queue.

        Returns:
            An async iterator of QueueItems.

        """
        yield QueueItem(EndEmpty(), AncestryTag())
        raise NotImplementedError

    async def start(self, tasks, collected_data: list[CollectedItem]):
        """Starts the run process on this node and all children.
        
        Arguments:
            tasks: A set of tasks.
            collected_data: An array to add collected items. 
        """
        async with self._start_lock:
            if not self._started:
                #self.print("STARTED NODE")
                task = asyncio.create_task(self.run(collected_data))
                tasks.add(task)
                task.add_done_callback(tasks.remove)
                self._started = True
        for name in self._children:
            await self._children[name].start(tasks, collected_data)

    @abstractmethod
    async def run(self, collected_data: list[CollectedItem]):
        """Runs the node.

        Pops item off of the queue and processes it. If the stop event is set,
        the node stops processing. For each item emitted from the
        process_queue_item method, the node pushes it to its children.
        After processing and pushing data to children, the node pops all data
        off the collected_data array and appends it to the collected_data
        array.

        Children should implement to define how exactly they want to process
        items off of the queues (e.g., either aysnchronously or synchronously).


        Arguments:
            collected_data: An array to add collected items.
        """
        raise NotImplementedError

    async def clean_up_end_state(self, collected_data: list[CollectedItem]):
        """Cleans up the end state of the node.
        
        Can be overwritten for any custom behavior.
        """
        return

    async def push_to_children(self, item: QueueItem):
        """Pushes item to all children.

        Arguments:
            item: The item to push to children, of type QueueItem.
        """
        tasks = set()
        async with asyncio.TaskGroup() as tg:
            for _child_key, child in self.get_children().items():
                if len(self._children) > 1:
                    # in cases where there is more than 1 child,
                    # need to deepcopy
                    to_push = copy.deepcopy(item)
                else:
                    to_push = item
                task = tg.create_task(child.put(self.key, to_push))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
        await asyncio.gather(*tasks)

class InOrderAsyncTreeNode(AsyncTreeNode):
    def __init__(self, key: TreeNodeKey):
        super().__init__(key)
    """A node that processes items off of parent queues in order."""
    
    async def run(self, collected_data: list[CollectedItem]):
        #self.print("At top of run function")
        while not(self._stop_event.is_set()):
            #self.print("At next loop of in order async tree node")
            next_dict = await self.get_next(collected_data)
            for parent_key, next_item in next_dict.items():
                #self.print(f"Got {next_item} from {parent_key}")
                if next_item.get_type() == QueueItemType.NO_OP:
                    continue
                async for item in self.process_queue_item(next_item, 
                                                          parent_key,
                                                          collected_data):
                    #self.print(f"Pushing to children: {item}")
                    await self.push_to_children(item)
        # push end to children after parent stop event is finished
        child_keys = [(c.key, c.uid) for c in self._children]
        #my_str = f"{Fore.RED}Node[{str(self), str(self.key)}]: "\
        # f"{Style.RESET_ALL}REACHED_END ITERATION; pushing end to "\
        # f"{child_keys}"
        #print(my_str)
        await self.clean_up_end_state(collected_data)
        #if "word_splitter" in self.key.key:
        #    import pdb; pdb.set_trace()
        await self.push_to_children(QueueItem(EndEmpty(), AncestryTag()))


    @abstractmethod
    async def process_queue_item(self, 
                                 next_item: QueueItem,
                                 parent_key: TreeNodeKey,
                                 collected_data: list[CollectedItem]
                                 ) -> AsyncIterator[QueueItem]:
        """Processes the next item on the queue.

        Arguments:
            next_item: The next item on the queue.

        Returns:
            An async iterator of QueueItems.

        """
        yield QueueItem(EndEmpty(), AncestryTag())
        raise NotImplementedError

class OutOfOrderAsyncTreeNode(AsyncTreeNode):
    def __init__(self, key: TreeNodeKey):
        super().__init__(key)

    async def process_single_queue_item(self, 
                                        item: QueueItem,
                                        parent_key: TreeNodeKey,
                                        collected_data: list[CollectedItem]
                                        ):
        """
        This function processes a single queue item, and yields the results to
        the children. It safely appends any collected data to the global
        collected data array with the lock, which may be accessed
        asynchronously by other tasks.
        """
        local_collected_data: list[CollectedItem] = []
        async for item in self.process_queue_item(item,
                                                  parent_key,
                                                  local_collected_data):
            await self.push_to_children(item)
        if len(local_collected_data) > 0:
            async with self._collected_data_lock:
                collected_data.extend(local_collected_data)
    
    
    """A node that processes items off of parent queues out of order."""
    async def run(self, collected_data: list[CollectedItem]):
        """Process items off of the parent queue asynchronously."""
        tasks = set()
        #self.print("At top of run function")
        while not (self._stop_event.is_set()):
            next_dict = await self.get_next(collected_data)
            for parent_key, next_item in next_dict.items():
                if next_item.get_type() == QueueItemType.NO_OP:
                    continue
                async_task = asyncio.create_task(
                    self.process_single_queue_item(next_item,
                                                   parent_key,
                                                   collected_data))
                tasks.add(async_task)
                async_task.add_done_callback(tasks.remove)
                # ensure tasks are given time to sleep
        if tasks:
            #self.print(f"Waiting for {len(tasks)} tasks to finish")
            # should only yield END to children when all tasks are done actually
            await asyncio.wait(tasks)
            #self.print(f"Finished waiting for tasks")
        # only push done when all invoked process tasks are done
        await self.clean_up_end_state(collected_data)
        await self.push_to_children(QueueItem(EndEmpty(), AncestryTag()))
        #self.print(f"Finished processing; pushed end to children")
        

    @abstractmethod
    async def process_queue_item(self, 
                                 next_item: QueueItem,
                                 parent_key: TreeNodeKey,
                                 collected_data: list[CollectedItem]
                                 ) -> AsyncIterator[QueueItem]:
        """Processes the next item on the queue.

        Arguments:
            next_item: The next item on the queue.

        Returns:
            An async iterator of QueueItems.

        """
        yield QueueItem(EndEmpty(), AncestryTag())
        raise NotImplementedError

PartialOutputType = TypeVar('PartialOutputType', bound = "LMPartialOutput")
PartialOutputTypeB = TypeVar('PartialOutputTypeB', bound = "LMPartialOutput")

class Splitter(Generic[PartialOutputType], InOrderAsyncTreeNode):
    """
    A node that splits text into parts based on a PartialOutputType.

    The PartialOutputType is a class that defines the splitting pattern of the
    emitted text; it has the following attributes:
        - delimeter_tokens: A list of delimeter tokens.
        - splitting_pattern: A regex pattern for splitting text.
        - delimeter_name: A name for the delimeter.

    Attributes:
        _output_type: The type of the output, of type PartialOutputType.
        _output_type_instance: An instance of the output type, containing an
        empty string.
    """
    def __init__(self, key: TreeNodeKey, 
                 output_type: Type[PartialOutputType]):
        super().__init__(key)
        self._output_type = output_type
        self._output_type_instance = output_type("")

    def __str__(self):
        return "Splitter[{}]".format(str(self._output_type_instance))

    async def process_queue_item(self, 
                                 next_item: QueueItem,
                                 parent_key: TreeNodeKey,
                                 collected_data: list[CollectedItem]
                                 ) -> AsyncIterator[QueueItem]:
        """Processes the next item on the queue.

        If the next item is of type PARTIAL_TEXT, this function checks to see if
        the text contains the splitting pattern. If it does, the text is split
        and the parts before and after the delimeter are emitted as separate
        items, with an END_OF_PARTIAL_OUTPUT item in between. If the text does
        not contain the splitting item, the text is emitted as a single
        PARTIAL_TEXT item.

        If the next item is of type END_OF_PARTIAL_OUTPUT, this function emits
        an END_OF_PARTIAL_OUTPUT item and an extra END_OF_STREAM item, to
        indicate that a steam has ended and the next stream is starting.

        If the next item is of type END_OF_STREAM, this function emits an
        END_OF_STREAM item.

        Arguments:
            next_item: The next item on the queue.

        Returns:
            An async iterator of QueueItems, to be pushed to children.

        Raises:
            RuntimeError: If the next item is not of type PARTIAL_TEXT,
            END_OF_PARTIAL_OUTPUT, or END_OF_STREAM.

        """
        if next_item.item.get_type() == QueueItemType.CTRL_MESSAGE:
            yield next_item
        elif next_item.item.get_type() == QueueItemType.END_OF_PARTIAL_OUTPUT:
            end_of_stream_item = QueueItem(
                    EndofStreamEmpty(), 
                    copy.deepcopy(next_item.ancestry)
                    )
            # we are yielding the partial first...
            #self.print("Received partial end")
            yield next_item
            # insert extra end of stream as parent has yielded partial
            yield end_of_stream_item
        elif next_item.item.get_type() == QueueItemType.END_OF_STREAM:
            yield next_item
        elif next_item.item.get_type() == QueueItemType.PARTIAL_TEXT:
            ancestry = next_item.ancestry
            text = next_item.item.get_inner()
            if self.delimeter_tokens != [""]:
                search_res = re.search(self.splitting_pattern, text)
            else:
                search_res = None
            if search_res is None:
                yield next_item
            else:
                parts = re.split(self.splitting_pattern, text)
                for i, part in enumerate(parts):
                    queue_item = QueueItem(PartialTextItem(part),
                                           copy.deepcopy(ancestry))
                    yield queue_item
                    if i != len(parts) - 1:
                        yield QueueItem(EndOfPartialEmpty(), copy.deepcopy(ancestry))
        else:
            raise RuntimeError(f"[Splitter [{self.delimeter_name}]] Received bad input type: {next_item.item.get_type()}")

    @property
    def delimeter_tokens(self):
        """A list of delimeter tokens."""
        return self._output_type_instance.delimeter_tokens

    @property
    def splitting_pattern(self):
        """A regex pattern for splitting text."""
        return self._output_type_instance.splitting_pattern

    @property
    def delimeter_name(self):
        """A name for the delimeter."""
        return self._output_type_instance.delimeter_name
    
class TextPipePmapEntry(Generic[PartialOutputType], InOrderAsyncTreeNode):
    """A node that processes text-based queue items and modifies ancestry.

    Attributes:
        _cnt: The count of partial outputs in the current stream.
        _is_last: A boolean indicating whether the partial output is the last.
        _last_ancestry: The ancestry of the last queue item seen.
        _output_type_instance: An instance of the output type this node
        processes,cotaining an empty string.
        _buffered_text: A string of text that has been buffered. This is
        necessary to make sure empty strings are not part of the count.
    """

    def __init__(self, 
                 key: TreeNodeKey,
                 output_type: Type[PartialOutputType],
                 scope_id: int):
        super().__init__(key) 
        self._cnt: int = 0
        self._is_last = False
        self._last_ancestry = AncestryTag()
        self._output_type_instance = output_type("")
        self._buffered_text = ""
        self._scope_id = scope_id

    def set_partial_ended(self):
        """Sets the partial output to ended."""
        self._cnt += 1
        self._last_ancestry = AncestryTag()

    @property
    def stream_ended(self):
        """A boolean indicating whether a stream has ended."""
        return self._is_last == False and self._cnt == 0

    def set_stream_ended(self):
        """Sets the stream to ended."""
        self._last_ancestry = AncestryTag()
        self._cnt = 0
        self._is_last = False

    def __str__(self):
         return f"TextPipePmapEntry[{str(self._output_type_instance)}]"

    def get_new_ancestry(self, original_ancestry: AncestryTag):
        """Returns a new ancestry by adding a tag to given ancestry.

        Arguments:
            original_ancestry: The original ancestry, of type AncestryTag.
        """
        return original_ancestry.push(AncestryId(self._cnt, self._scope_id))

    async def process_queue_item(self,
                                 next_item: QueueItem,
                                 parent_key: TreeNodeKey,
                                 collected_data: list[CollectedItem]
                                 ) -> AsyncIterator[QueueItem]:
        """Processes the next item on the queue.

        If the next item is of type PARTIAL_TEXT, this function buffers the
        text, and yields the same item with a modified ancestry based on the
        current state.

        If the next item is of type END_OF_PARTIAL_OUTPUT, this function emits
        an END_OF_PARTIAL_OUTPUT item with modified ancestry, and sets the
        partial output to ended, if there is non-empty buffered text.

        If the next item is of type END_OF_STREAM, this function emits an
        END_OF_STREAM item with modified ancestry based on current state, and
        sets the stream to be ended.

        Arguments:
            next_item: The next item on the queue.

        Returns:
            An async iterator of QueueItems.

        Raises:
            RuntimeError: If the next item is not of type PARTIAL_TEXT,
            END_OF_PARTIAL_OUTPUT, or END_OF_STREAM.

            AssertionError: If the ancestry of the next item is not as the same
            as the last ancestry received, when the last ancestry is non-empty.
        """
        

        #self.print(f"Received queue item: "\
        #                f" {str(next_item.item)} with"\
        #                f" ancestry: {next_item.ancestry},"\
        #                f" current cnt: {self._cnt},"\
        #                f" buffered: {self._buffered_text},"\
        #                f" is_last: {self._is_last}")
        if next_item.item.get_type() == QueueItemType.CTRL_MESSAGE:
            yield next_item
        else:
            if not self._last_ancestry.is_empty():
                if next_item.ancestry != self._last_ancestry:
                    logging.warning(f"[{self.key}] ancestry match bad")
                    logging.warning(f"[{self.key}] last: "\
                    f"{self._last_ancestry}, next: {next_item.ancestry}")
                assert next_item.ancestry == self._last_ancestry, f"[{str(self)}] last: {self._last_ancestry}, next: {next_item.ancestry}"
            self._last_ancestry = next_item.ancestry
            if next_item.item.get_type() == QueueItemType.END_OF_STREAM:
                # if stream already ended, do nothing
                if not self.stream_ended:
                    self._is_last = True
                    # will still increment the count
                    next_item.ancestry = self.get_new_ancestry(next_item.ancestry)
                    yield next_item
                self.set_stream_ended()
            elif next_item.item.get_type() == QueueItemType.END_OF_PARTIAL_OUTPUT:
                next_item.ancestry = self.get_new_ancestry(next_item.ancestry)
                yield next_item
                if self._buffered_text != '':
                    self.set_partial_ended()
                    self._buffered_text = ''
            elif next_item.item.get_type() == QueueItemType.PARTIAL_TEXT:
                self._buffered_text += next_item.item.get_inner()
                self._last_item = next_item.item.get_inner()
                next_item.ancestry = self.get_new_ancestry(next_item.ancestry)
                yield next_item
            else:
                raise RuntimeError(f"[PmapEntry] Received bad input type: {next_item.item.get_type()}")

class TextPipeResolver(Generic[PartialOutputType], InOrderAsyncTreeNode):
    """A node that resolves text-based queue items.
    Attributes:
        _output_type: The type of the output, of type PartialOutputType.
        _output_type_instance: An instance of the output type, containing an
        empty string.
        _current_ancestry: The last ancestry value received, used for debug
        checks.
        _buffered: Buffered text.

    """
    def __init__(self, 
                 key: TreeNodeKey,
                 output_type: Type[PartialOutputType]):
        super().__init__(key) 
        self._output_type = output_type
        self._output_type_instance = output_type("")
        self._current_ancestry = AncestryTag()
        self._buffered = ""
        self._emit_eos = True
        self._last_emitted_eos = False

    @property
    def emit_eos(self):
        return self._emit_eos

    @emit_eos.setter
    def emit_eos(self, v):
        self._emit_eos = v
    
    def __str__(self):
        return "TextPipeResolver[{}]".format(str(self._output_type_instance))

    async def emit_buffered(self) -> AsyncIterator[QueueItem]:
        """Emits any buffered text as an iterator of QueueItems.

        The emitted items are of type ResolvedText.
        """
        if len(self._buffered) > 0:
            item = QueueItem(ResolvedText(self._output_type(self._buffered)), copy.deepcopy(self._current_ancestry))
            yield item

    async def process_queue_item(self, 
                                 next_item: QueueItem,
                                 parent_key: TreeNodeKey,
                                 collected_data: list[CollectedItem]
                                 ) -> AsyncIterator[QueueItem]:
        """Processes the next item on the queue.

        If the next item is of type PARTIAL_TEXT, this function buffers the text
        and emits nothing.

        If the next item is of type END_OF_PARTIAL_OUTPUT, this function emits
        the current buffered text as a ResolvedText and sets the buffered text
        to be empty.

        If the next item is of type END_OF_STREAM, this function emits the
        current buffered text as a ResolvedText and sets the buffered text to be
        empty.

        Attributes:
            next_item: The next item on the queue, of type QueueItem.

        Returns:
            An async iterator of QueueItems.

        Raises:
            RuntimeError: If the next item is not of type PARTIAL_TEXT,
            END_OF_PARTIAL_OUTPUT, or END_OF_STREAM.

            AssertionError: If the ancestry of the next item is not the same as
            the last ancestry received, when the last ancestry is non-empty.

        """
        self.print(f"On key {self.key} processing {next_item.item} with ancestry {next_item.ancestry}")
        
        if next_item.item.get_type() == QueueItemType.CTRL_MESSAGE:
            yield next_item
        
        else:
            if not self._current_ancestry.is_empty():
                if next_item.ancestry != self._current_ancestry:
                    logging.warning(f"[{self.key}] ancestry match bad")
                    logging.warning(f"[{self.key}] last: "\
                    f"{self._current_ancestry}, next: {next_item.ancestry}")
                if self._current_ancestry != next_item.ancestry:
                    raise ValueError(
                        f"[{str(self)}] last ancestry: {self._current_ancestry}, next ancestry: {next_item.ancestry}, "
                        f"next item: {next_item.item}, on key "
                    )
            self._current_ancestry = next_item.ancestry
            
            if next_item.item.get_type() == QueueItemType.END_OF_STREAM:
                async for item in self.emit_buffered():
                    yield item
                    self._last_emitted_eos = False
                self._current_ancestry = AncestryTag()
                if self._emit_eos and not self._last_emitted_eos:
                    self.print(f"Yielding EOS with {next_item.ancestry}")
                    yield next_item
                    self._last_emitted_eos = True
            elif next_item.item.get_type() == QueueItemType.END_OF_PARTIAL_OUTPUT:
                async for item in self.emit_buffered():
                    self.print(f"Yielding {item}")
                    yield item
                    self._last_emitted_eos = False
                self._buffered = ""
                self._current_ancestry = AncestryTag()
            elif next_item.item.get_type() == QueueItemType.PARTIAL_TEXT:
                self._current_ancestry = next_item.ancestry
                self._buffered += next_item.item.get_inner()
            else:
                raise RuntimeError(f"[Resolver] Received bad input type: {next_item.item.get_type()}")

class DebugNode(InOrderAsyncTreeNode):
    """A node that prints debugging information."""
    def __init__(self, key: TreeNodeKey, name: str):
        super().__init__(key)
        self._name = name
        self._ignore_control = True

    @property
    def ignore_control(self):
        return self._ignore_control

    @ignore_control.setter
    def ignore_control(self, val):
        self._ignore_control = val

    def __str__(self):
        return f"DebugNode[{self._name}]"

    async def process_queue_item(self, 
                                 next_item: QueueItem,
                                 parent_key: TreeNodeKey,
                                 collected_data: list[CollectedItem],
                                 ) -> AsyncIterator[QueueItem]:
        """Processes the next item on the queue.

        This function collects the next item and yields it on to its children.

        Arguments:
            next_item: The next item on the queue.

        Returns:
            An async iterator of QueueItem
        """
        # ignore the message if it is a control message
        if next_item.get_type() == QueueItemType.CTRL_MESSAGE and self._ignore_control:
            return
        collected_data.append(CollectedItem(str(self), next_item))
        yield next_item

    async def clean_up_end_state(self, collected_data: list[CollectedItem]):
        async with self._collected_data_lock:
            collected_data.append(CollectedItem(str(self), QueueItem(EndEmpty(),
                                                                 AncestryTag())))

class NoOpNode(InOrderAsyncTreeNode):
    """A node that does nothing."""
    def __init__(self, key: TreeNodeKey):
        super().__init__(key)

    def __str__(self):
        return f"NoOpNode{str(self.key)}"

    async def process_queue_item(self, 
                                 next_item: QueueItem,
                                 parent_key: TreeNodeKey,
                                 collected_data: list[CollectedItem]
                                 ) -> AsyncIterator[QueueItem]:
        """Processes the next item on the queue.

        This function yields the next item on to its children.

        Arguments:
            next_item: The next item on the queue.

        Returns:
            An async iterator of QueueItem
        """
        yield next_item

async def add_to_queue(node: AsyncTreeNode, stream:
                       AsyncIterator[str], with_wait = False):
    """Adds a stream to the input queue of a node.

    Arguments:
        node: The node to add the stream to.
        stream: An async iterator of strings to add.
        with_wait: Whether to sleep in between adding items. Main use is for
        tests.

    """
    root_queue = asyncio.Queue()
    node.add_parent_queue(TreeNodeKey("root"), root_queue)
    async for item in stream:
        queue_item = QueueItem(PartialTextItem(item), AncestryTag())
        await root_queue.put(queue_item)
        await asyncio.sleep(0.01)
    end_partial = QueueItem(EndOfPartialEmpty(), AncestryTag())
    end_empty = QueueItem(EndEmpty(), AncestryTag())
    await root_queue.put(end_partial)
    await root_queue.put(end_empty)
class AncestryLevel(object):
    """Represents how many levels of ancestry to remove when considering
    grouping an ancestry.

    If level is 0, then consider the ancestry tag as is.
    If level is 1, consider the ancestry tag with the last level removed.
    """

    def __init__(self, level: int):
        self._level = level

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AncestryLevel):
            return False
        return self._level == other._level

    @property
    def level(self):
        return self._level

    def __hash__(self):
        return hash(self._level)

    def __str__(self):
        return f"AncestryLevel({self._level})"

    def get_aggregated_ancestry(self, ancestry: AncestryTag) -> AncestryTag:
        """Returns the ancestry tag with self._level levels removed.

        Also removes the "is_eof" flag, so the ancestry tag can be used to index
        into a hashmap.
        """
        new_ancestry = copy.deepcopy(ancestry)
        for i in range(self._level):
            new_ancestry, _ = new_ancestry.pop()
        new_ancestry.is_eos = False
        return new_ancestry


class QueueRule(object):
    """Associates a queue with a particular ancestry level."""

    def __init__(self,
                 queue_key: TreeNodeKey,
                 ancestry_level: AncestryLevel):
        self._queue_key = queue_key
        self._ancestry_level = ancestry_level

    @property
    def queue_key(self):
        return self._queue_key

    @property
    def ancestry_level(self):
        return self._ancestry_level

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QueueRule):
            return False
        return self._queue_key == other._queue_key and self._ancestry_level == other._ancestry_level

    def __hash__(self):
        return hash((self._queue_key, self._ancestry_level))


class Rule(object):
    """
    Rule that governs how data coming on input queues relate to data going out
    of output queues.

    For example, in the case where the pipeline has a stream collector,
    multiple inputs corresponding to a single output, all across the same
    ancerstry.
    In cases where the outgoing data is first split and then pmap'd, then,
    there would be multiple outputs corresponding to a single input (or set of
    inputs).
    Different output streams may also have additional levels of ancestries.
    """

    def __init__(self,
                 recv_queue_rules: list[QueueRule],
                 output_queue_rules: list[list[QueueRule]],
                 ):
        """
        Initializes a rule.
            recv_queue_rules: list of receive queues and their associated
        ancestry levels to group together.
            output_queue_rules: list of output queues and their associated
            ancestry levels to group together.
            It is a nested list due to conditionals.
            The outer list represents the OR while the inner list
            represents the AND.
        """

        self._recv_queue_map = {}
        self._output_queue_map = {}
        for rule in recv_queue_rules:
            # logging.info(f"Adding rule {rule.queue_key} with level {rule.ancestry_level}")
            self._recv_queue_map[rule.queue_key] = rule.ancestry_level
        for outer_list in output_queue_rules:
            for rule in outer_list:
                # logging.info(f"Adding rule {rule.queue_key} with level {rule.ancestry_level}")
                self._output_queue_map[rule.queue_key] = rule.ancestry_level
        self._output_queue_rules = output_queue_rules
    @property
    def output_queue_rules(self):
        return self._output_queue_rules

    @property
    def send_queues_map(self):
        return self._output_queue_map

    @property
    def recv_queues_map(self):
        return self._recv_queue_map

    def in_recv_queues(self, key: TreeNodeKey) -> bool:
        return key in self._recv_queue_map

    def in_send_queues(self, key: TreeNodeKey) -> bool:
        return key in self._output_queue_map

    def get_ancestry_level_recv(self, key: TreeNodeKey) -> AncestryLevel:
        return self._recv_queue_map[key]

    def get_ancestry_level_send(self, key: TreeNodeKey) -> AncestryLevel:
        return self._output_queue_map[key]


def get_raw_queue_name(key: TreeNodeKey) -> str:
    if key.key.endswith("_send"):
        return key.key[:-5]
    elif key.key.endswith("_recv"):
        return key.key[:-5]
    else:
        raise ValueError(f"Key {key} is not a send or recv node")


class EventType(Enum):
    START_PROCESSING = 0  # Start processing
    RECEIVED_DATA = 1  # Data received from a queue
    SENT_DATA = 2  # Data sent to a queue
    END_PROCESSING = 3  # Received end processing
    LM_DATA = 4 # Prompt/response received from LM

class StageProcessingLogEvent:
    def __init__(self,
                 ancestry: AncestryTag,
                 event_type: EventType,
                 queue_name: TreeNodeKey):
        self._ancestry = ancestry
        self._event_type = event_type
        self._queue_name = queue_name
        self._timestamp = time.time()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StageProcessingLogEvent):
            return False
        # note; this does NOT check for timestamp equality
        return self._ancestry == other._ancestry and self._event_type == other._event_type and self._queue_name == other._queue_name

    @property
    def ancestry(self):
        return self._ancestry

    @property
    def event_type(self):
        return self._event_type

    @property
    def queue_name(self):
        return self._queue_name

    @property
    def timestamp(self):
        return self._timestamp

    def __str__(self):
        return f"[{self._ancestry}] Event {self._event_type} on {self._queue_name} at {self._timestamp}"


class AltoRequestProcessingEvent(QueueObject):
    def __init__(
        self: Self,
        base_ancestry: AncestryTag,
        rule: Rule,
        stage_id: int,
        replica_id: int,
    ) -> None:

        self._ancestry_key = base_ancestry
        self._received_data: list[StageProcessingLogEvent] = []
        self._sent_data: list[StageProcessingLogEvent] = []
        self._stage_id = stage_id
        self._replica_id = replica_id
        self._end_time = None
        self._start_time = None
        self._lm_prompt = ""
        self._lm_response = ""
        self._received_end: list[dict[TreeNodeKey, bool]]  = []
        for send_rule in rule.output_queue_rules:
            new_dict = {}
            for send_key in send_rule:
                new_dict[send_key.queue_key] = False
            self._received_end.append(new_dict)
        self._rule = rule

    @property
    def rule(
        self: Self
    ):
        return self._rule

    @property
    def ancestry(self):
        return self._ancestry_key

    def debug_print(self) -> str:
        ret = f"[({self._stage_id}, {self._replica_id})i"\
            f"{self._ancestry_key}]"

        ret += "\nReceived Data:\n"
        for event in self._received_data:
            ret += f"\t{event}\n"
        ret += "Sent Data:\n"
        for event in self._sent_data:
            ret += f"\t{event}\n"

        ret += f"End Time: {self._end_time}\n"
        return ret

    def get_csv_strings(self) -> list[str]:
        ret = []
        base_format_string = f"{self._stage_id},"\
            f"{self._replica_id},"\
            f"{self._ancestry_key.csv_str()},"
        for event in self._received_data:
            new_format_string = base_format_string + \
                f"{event.ancestry.csv_str()},"\
                f"{get_raw_queue_name(event.queue_name)},"\
                f"{event.timestamp},"\
                "RECV"
            ret.append(new_format_string)
        for event in self._sent_data:
            new_format_string = base_format_string + \
                f"{event.ancestry.csv_str()},"\
                f"{get_raw_queue_name(event.queue_name)},"\
                f"{event.timestamp},"\
                "SENT"
            ret.append(new_format_string)
        if self._start_time is not None:
            new_format_string = base_format_string + \
                f"{[]},"\
                f"None,"\
                f"{self._start_time},"\
                "START,"\
                f"{self._lm_prompt},"\
                f"{self._lm_response}"
            ret.append(new_format_string)

        if self._end_time is not None:
            new_format_string = base_format_string + \
                f"{[]},"\
                f"None,"\
                f"{self._end_time},"\
                "END,"\
                f"{self._lm_prompt},"\
                f"{self._lm_response}"
            ret.append(new_format_string)
        # in buggy cases, end time may not be recorded
        # else:
            # raise ValueError("End time not recorded")
        return ret

    def record_lm_msg(
        self: Self,
        queue_name: TreeNodeKey,
        ts: float,
        lm_prompt: str,
        lm_response: str
    ) -> None:
        self._lm_prompt = lm_prompt
        self._lm_response = lm_response

    def record_starting(
        self: Self,
        queue_name: TreeNodeKey,
        ts: float
    ) -> None:
        self.record_start(ts)

    def record_finished(
        self: Self,
        queue_name: TreeNodeKey,
        ts: float
    ) -> bool:
        for i, send_rule in enumerate(self._rule.output_queue_rules):
            for send_key in send_rule:
                if queue_name == send_key.queue_key:
                    self._received_end[i][send_key.queue_key] = True
                    # check if all send queues have been received
                    if all(self._received_end[i].values()):
                        self.record_end(ts)
                        return True
                    break
        return False

    @property
    def sent_data(self):
        return self._sent_data

    @property
    def received_data(self):
        return self._received_data

    @property
    def finished_map(self):
        return self._received_end

    def get_type(self) -> QueueItemType:
        return QueueItemType.LOG_MSG

    def add_received_data(self, event: StageProcessingLogEvent):
        self._received_data.append(event)

    def add_sent_data(self, event: StageProcessingLogEvent):
        self._sent_data.append(event)

    def record_start(self, ts):
        """
        If state node gets *multiple* start events for an ancestry,
        it will overwrite the last one.
        """
        self._start_time = ts

    def record_end(self, ts: float):
        self._end_time = ts

    def calculate_finished_processing(self) -> list[FinishedProcessing]:
        decrement_map = collections.defaultdict(int)
        for event in self._received_data:
            decrement_map[event.queue_name] += 1
        return [FinishedProcessing(incoming_queue_name=get_raw_queue_name(key),
                                   count=value) for key, value in
                decrement_map.items()]

    def alto_metadata_bytes(self) -> bytes:
        # serialize ancestry tag
        ancestry_proto = self._ancestry_key.to_proto()
        ancestry_proto = self.assign_alto_controller_event(
            ancestry_proto,
            self._stage_id,
            self._replica_id
        )
        ancestry_proto.sticky_routing_hash = self._ancestry_key.sticky_routing_hash
        return ancestry_proto.SerializeToString()

    def calculate_time(self):
        processing_time = self._end_time - self._received_data[0]._timestamp
        return processing_time

    def assign_alto_controller_event(self,
                                     proto,
                                     stage_id: int,
                                     replica_id: int):
        proto.finished_processing_log_item.stage_id = stage_id
        proto.finished_processing_log_item.replica_id = replica_id
        proto.finished_processing_log_item.finished_processing.extend(
            self.calculate_finished_processing()
        )
        if self._end_time is None:
            # could be case of no send and received events
            if len(self._received_data) == 0 and len(self._sent_data) == 0:
                proto.finished_processing_log_item.request_processing_seconds = 0
                return proto
            raise ValueError("End time not recorded")
        if self._start_time is not None:
            processing_time = self._end_time - self._start_time 
        else:
            processing_time = self._end_time - self._received_data[0]._timestamp
        proto.finished_processing_log_item.request_processing_seconds =\
            processing_time
        return proto

    def get_inner(self) -> Any:
        # calculate the time taken to process the request
        pass


empty_processing_event = AltoRequestProcessingEvent(AncestryTag([]), Rule([], []), 0, 0)

async def main():
    """An example of a text processing pipeline."""
    async def lm_engine_example():
        prompt = "Hello world\nThis is an example\nOf a text stream\nextra text without newline"
        for char in prompt:
            yield char
            await asyncio.sleep(0.01)  # Simulate async text generation
        next_part = "\nNewline test\nactual last part"
        yield next_part

    root_node = Splitter[FullTextOutput](TreeNodeKey("full_splitter"), FullTextOutput)
    full_resolver = TextPipeResolver[FullTextOutput](TreeNodeKey("full_resolver"), FullTextOutput)
    line_node = Splitter[LineOutput](TreeNodeKey("line_splitter"),
                                          LineOutput)
    root_node.add_child(line_node)
    root_node.add_child(full_resolver)
    full_resolver.add_child(DebugNode(TreeNodeKey("debug_full_resolver"), "FullTextResolver"))
    
    line_pmap = TextPipePmapEntry(TreeNodeKey("line_pmap"),
                                  LineOutput,
                                  scope_id=1)
    line_node.add_child(line_pmap)
    line_resolver = TextPipeResolver[LineOutput](TreeNodeKey("line_resolver"), LineOutput)
    line_pmap.add_child(line_resolver)
    line_resolver.add_child(DebugNode(TreeNodeKey("debug_line_resolver"), "LineResolver"))
    word_node = Splitter[WordOutput](TreeNodeKey("word_splitter"), WordOutput)
    line_pmap.add_child(word_node)

    word_pmap = TextPipePmapEntry(TreeNodeKey("word_pmap"), WordOutput,
                                  scope_id=2)
    word_node.add_child(word_pmap)
    word_resolver = TextPipeResolver(TreeNodeKey("word_resolver"), WordOutput)
    word_pmap.add_child(word_resolver)
    word_resolver.add_child(DebugNode(TreeNodeKey("debug_word_resolver"), "WordResolver"))

    tasks = set()
    collected_items = []

    await root_node.start(tasks, collected_items)
    await add_to_queue(root_node,
                        lm_engine_example())

    await asyncio.gather(*tasks)
    for item in collected_items:
        print(f"{item.putter}: {item.item}")
    
if __name__ == '__main__':
    asyncio.run(main())

