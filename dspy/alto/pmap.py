import asyncio
from typing import AsyncIterator, Tuple, Set
from alto.ancestry import AncestryId
from alto.base import QueueItem, EndofStreamEmpty
from typing import Dict
async def pmap(
    input_stream: AsyncIterator[QueueItem],
    target_module,
    scope_id: int = 2,
) -> AsyncIterator[QueueItem]:
    """
    Add ancestry, run stream of QueueItem through a module in parallel,
    while preserving input order.
    """
    out_q: asyncio.Queue[Tuple[int, QueueItem]] = asyncio.Queue()
    tasks: Set[asyncio.Task] = set()
    started = 0

    async def _run(submit_idx: int, input_item: QueueItem):
        cur_item = input_item.item
        cur_ancestry = input_item.ancestry

        # Add ancestry marker for this call
        pushed = cur_ancestry.push(AncestryId(submit_idx, scope_id))
        print("---pmap added ancestry")

        try:
            result = await target_module.acall(cur_item)
            if isinstance(result, EndofStreamEmpty):
                return

            # Remove ancestry marker
            parent, _ = pushed.pop()
            print("---pmap removed ancestry")

            await out_q.put((submit_idx, QueueItem(result, parent)))
        except Exception as e:
            print(f"pmap failed with error message: {e}")
            raise

    # Spawn tasks for each input
    async for qitem in input_stream:
        task = asyncio.create_task(_run(started, qitem))
        tasks.add(task)
        started += 1

    # Collect results in order
    next_idx = 0
    buffer: Dict[int, QueueItem] = {}
    received = 0

    try:
        while received < started:
            i, res = await out_q.get()
            received += 1
            buffer[i] = res

            while next_idx in buffer:
                yield buffer.pop(next_idx)
                next_idx += 1
    except Exception as e:
        print(f"pmap failed with error message: {e}")
        for task in list(tasks):
            task.cancel()
        raise
    finally:
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)