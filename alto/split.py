import asyncio

from alto.ancestry import  AncestryTag
from alto.base import ( 
    QueueItemType, CtrlMessageType,  QueueItem, PartialTextItem,EndOfPartialEmpty, EndofStreamEmpty, EndEmpty
)
from alto.queue import (
     OutputQueue, CtrlMessage
)
from alto.lmtextpipe import LMPartialOutput, FullTextOutput, SentenceOutput, LineOutput, WordOutput
import asyncio
from typing import AsyncGenerator, Optional, Any, List, Type
from alto.nodes import (
    Splitter, TextPipeResolver, TreeNodeKey,
    AsyncTreeNode, InOrderAsyncTreeNode, CollectedItem
)
import dspy
from dspy.streaming.messages import StreamResponse
from dspy.primitives.prediction import Prediction

# Add paths for your local alto_py library
# sys.path.append("/home/nmodugul/Desktop/lm-pipeline-serving/alto_py")
# sys.path.append("/home/nmodugul/Desktop/lm-pipeline-serving/alto_py/alto_lib")
from typing import Type
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

class OutputSplit:
    output_field: str               
    split_type: Type[SplitType]

class ConsumerObject:
    def __init__(self, name: str, granularity: Any, output_split: Any, uid: int = 0):
        self.name = name
        self.granularity = granularity
        self.uid = uid

        self.output_queue = OutputQueue(f"consumer_{self.name}_{uid}")
        self.send_node = SendNode(TreeNodeKey(f"{granularity.__name__}_send", uid), self.output_queue)
        self.output_split = output_split ## Split passed to StreamingTextProducer

        # Construct Graph
        if granularity is self.output_split:
            # Same granularity: just resolve -> send
            self.resolver = TextPipeResolver(TreeNodeKey(f"{granularity.__name__}_resolver", uid), granularity)
            self.resolver.add_child(self.send_node)
            self.root_node = self.resolver
        else:
            # Finer granularity: splitter -> resolver -> send
            self.split_node = Splitter[granularity](TreeNodeKey(f"{granularity.__name__}_splitter", uid), granularity)
            self.resolver = TextPipeResolver(TreeNodeKey(f"{granularity.__name__}_resolver", uid), granularity)
            self.split_node.add_child(self.resolver)
            self.resolver.add_child(self.send_node)
            self.root_node = self.split_node

        self._feeder_key = TreeNodeKey(f"consumer_{self.name}_feeder", uid)
        self._feeder_queue = None
        self._graph_started = False

    async def _ensure_graph_started(self):
        if not self._graph_started:
            tasks = set()
            collected = []
            await self.root_node.start(tasks, collected)
            self._feeder_queue = asyncio.Queue()
            self.root_node.add_parent_queue(self._feeder_key, self._feeder_queue)
            self._graph_started = True

    async def process_queue_item(self, queue_item: QueueItem):
        await self._ensure_graph_started()
        await self._feeder_queue.put(queue_item)


# ----------------- Magic Pipe ----------------- #

class MagicTextPipe:
    def __init__(self, ancestry: AncestryTag, consumers: List[ConsumerObject]):
        self.ancestry = ancestry
        self.consumers = consumers
        self.queue_items: List[QueueItem] = []

    def add_queue_item(self, queue_item: QueueItem):
        self.queue_items.append(queue_item)

    def push_to_consumers(self, queue_item: QueueItem):
        self.add_queue_item(queue_item)
        for consumer in self.consumers:
            # schedule consumer processing (they will put it into their own internal queues)
            asyncio.create_task(consumer.process_queue_item(queue_item))

    def get_consumer_output(self, consumer_name: str) -> Optional[OutputQueue]:
        for consumer in self.consumers:
            if consumer.name == consumer_name:
                return consumer.output_queue
        return None

class SendNode(InOrderAsyncTreeNode):
    """Send data from the graph into an OutputQueue"""
    def __init__(self, key: TreeNodeKey, output_queue: "OutputQueue"):
        super().__init__(key)
        self.output_queue = output_queue

    async def process_queue_item(self, next_item, parent_key: TreeNodeKey, collected_data: List[CollectedItem]):
        # Write the object to the consumer output queue
        await self.output_queue.write_object(next_item.item, next_item.ancestry)

        # If this is an end-of-stream marker, close the consumer queue so
        # any async for loops over `.stream()` will finish.
        if isinstance(next_item.item, (EndofStreamEmpty, EndEmpty)):
            await self.output_queue.close()

        if False:  # keep generator interface compatibility
            yield

# ----------------- Streaming Producer ----------------- #

class StreamingTextProducer(dspy.Module):
    def __init__(self, signature: dspy.Signature, output_split: LMPartialOutput, consumers: List[ConsumerObject]):
        super().__init__()
        self.output_split = output_split
        if not signature.output_fields:
            raise ValueError(f"No outputs found for signature {signature}")
        stream_field = next(iter(signature.output_fields.keys()))
        self.input_fields = list(signature.input_fields.keys())
        self.generate = dspy.Predict(signature)
        listener = dspy.streaming.StreamListener(signature_field_name=stream_field)
        self.streamer = dspy.streamify(
            self.generate,
            is_async_program=True,
            async_streaming=True,
            stream_listeners=[listener]
        )
        self.consumers = consumers
        self.ancestry = None

    async def aforward(self, *args, **kwargs) -> AsyncGenerator[QueueItem, None]:
        # Map args/kwargs → inputs
        if args and not kwargs:
            if len(args) != len(self.input_fields):
                raise ValueError(
                    f"Got {len(args)} arguments, expected {len(self.input_fields)}: {self.input_fields}"
                )
            inputs = dict(zip(self.input_fields, args))
        else:
            inputs = dict(kwargs)

        # Build processing graph for this producer
        q = OutputQueue(f"queue_{self.output_split.__name__}")
        split_node = Splitter[self.output_split](
            TreeNodeKey(f"{self.output_split.__name__}_splitter", 0),
            self.output_split
        )
        send_node = SendNode(TreeNodeKey(f"{self.output_split.__name__}_send", 0), q)
        split_node.add_child(send_node)

        tasks = set()
        collected = []
        await split_node.start(tasks, collected)

        # Start feeder task
        feeder_task = asyncio.create_task(
            feed_stream_into_pipeline(
                self.streamer(**inputs),
                split_node,
            )
        )
        tasks.add(feeder_task)

        magic_text_pipe_obj: Optional[MagicTextPipe] = None

        # Consume output
        async for qitem in q.stream():
            self.ancestry = getattr(qitem, "ancestry", self.ancestry)

            if magic_text_pipe_obj is None:
                magic_text_pipe_obj = MagicTextPipe(ancestry=self.ancestry, consumers=self.consumers)

            if hasattr(qitem.item, "get_type") and qitem.item.get_type() == QueueItemType.END_OF_PARTIAL_OUTPUT:
                magic_text_pipe_obj.push_to_consumers(qitem)
                yield QueueItem(magic_text_pipe_obj, self.ancestry)
                magic_text_pipe_obj = None
            else:
                magic_text_pipe_obj.push_to_consumers(qitem)

        # Final flush (in case there are leftover items)
        if magic_text_pipe_obj and magic_text_pipe_obj.queue_items:
            yield QueueItem(magic_text_pipe_obj, self.ancestry)

async def feed_stream_into_pipeline(dspy_output_stream, root_node: AsyncTreeNode):
    ancestry = AncestryTag()
    root_queue = asyncio.Queue()
    feeder_key = TreeNodeKey("dspy_feeder")
    root_node.add_parent_queue(feeder_key, root_queue)

    await root_queue.put(QueueItem(CtrlMessage(CtrlMessageType.STARTING_PROCESSING), ancestry))

    saw_any_chunk = False
    final_prediction: Any = None
    try:
        async for item in dspy_output_stream:
            if isinstance(item, StreamResponse):
                saw_any_chunk = True
                await root_queue.put(QueueItem(PartialTextItem(item.chunk), ancestry))
            elif isinstance(item, Prediction):
                final_prediction = item
            else:
                raise TypeError(f"Unexpected item type: {type(item)}")
    finally:
        # if stream yielded no chunks but we have a final prediction, push it
        if not saw_any_chunk and final_prediction is not None:
            text = getattr(final_prediction, "response", "") or ""
            if text:
                await root_node.put(feeder_key, QueueItem(PartialTextItem(text), ancestry))

    # End of stream - send end markers into the graph
    await root_queue.put(QueueItem(EndOfPartialEmpty(), ancestry))
    await root_queue.put(QueueItem(EndofStreamEmpty(), ancestry))
    await root_queue.put(QueueItem(EndEmpty(), ancestry))

