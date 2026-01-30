# why can't Python just have a priority_queue data structure?

from typing import Self, TypeVar, Protocol, Sequence, Any
import heapq

T = TypeVar("T")

class SupportsRichComparison(Protocol):
    def __lt__(self, other: Any, /) -> bool: ...
    def __eq__(self, other: Any, /) -> bool: ...
    def __gt__(self, other: Any, /) -> bool: ...

class PriorityQueue[T: SupportsRichComparison]:
    def __init__(self, iterable: Sequence[T] = []) -> None:
        self.l = list(iterable)
        
    def __len__(self) -> int:
        return len(self.l) 
    
    def push(self, val: T) -> None:
        heapq.heappush(self.l, val)        
        
    def pop(self) -> T:
        return heapq.heappop(self.l)
