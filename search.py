from abc import abstractmethod
from priority_queue import PriorityQueue
from collections import deque
from typing import Dict, List, Tuple, Deque, Optional, Set
from copy import deepcopy # there's a story this module is here
import heapq
from math import hypot

class BaseSearch:
    # lol very tricky type hint
    def __init__(self, 
                 nodes: Dict[str, List[int]],
                 edges: Dict[str, Dict[str, int]], 
                 origin: str,
                 destinations: List[str]) -> None:
        # account for iterable nonsense
        self.nodes = deepcopy(nodes)
        self.edges = deepcopy(edges)
        self.origin = origin
        self.destinations = deepcopy(destinations)
        
    @abstractmethod
    def search(self) -> List[str]:
        """search _summary_
        
        Abstract Method for implementing search algorithms.

		Returns:
			SearchModel: _description_
		"""        
        return []
    
    def cost(self, path: List[str]) -> int:
        if len(path) == 0 or self.edges is None : return -1
        
        c = 0
        for start, end in zip(path[:-1], path[1:]):
            c += self.edges[start][end]
        return c
    
        
class DFSSearch(BaseSearch):
    def __init__(self, nodes: Dict[str, List[int]], edges: Dict[str, Dict[str, int]], origin: str, destinations: List[str]) -> None:
        super().__init__(nodes, edges, origin, destinations)
        
    def search(self) -> List[str]:
        """search _summary_
        
        Implement the DFS algorithm:

		Returns:
			SearchModel: _description_
		""" 
        stack: Deque[str] = deque([self.origin])
        parent: Dict[str, Optional[str]] = {self.origin: None}
        visited: Set[str] = {self.origin}
        
        while len(stack) > 0:
            current = stack.pop()
            if current in self.destinations:
                # reconstruct path
                path: List[str] = []
                cur: Optional[str] = current
                while cur is not None:
                    # traceback by parent
                    path.append(cur)
                    cur = parent.get(cur)
                path.reverse()
                return path
            
            for node in self.edges.get(current, {}):
                if node in visited:
                    continue
                visited.add(node)
                parent[node] = current
                stack.append(node)
        return []

class BFSSearch(BaseSearch):
    def __init__(self, nodes: Dict[str, List[int]], edges: Dict[str, Dict[str, int]], origin: str, destinations: List[str]) -> None:
        super().__init__(nodes, edges, origin, destinations)
        
    def search(self) -> List[str]:
        """search _summary_
        
        Implement the BFS algorithm:

		Returns:
			SearchModel: _description_
		""" 
        queue: Deque[str] = deque([self.origin])
        parent: Dict[str, Optional[str]] = {self.origin: None}
        visited: Set[str] = {self.origin}
        
        while len(queue) > 0:
            current = queue.popleft()

            # get current point. If it is a destination, return.
            if current in self.destinations:
                # reconstruct path
                path: List[str] = []
                cur: Optional[str] = current
                while cur is not None:
                    # traceback by parent
                    path.append(cur)
                    cur = parent.get(cur)
                path.reverse()
                return path
            
            # otherwise, keep searching.
            for nxt in self.edges.get(current, {}).keys():
                if nxt in visited:
                    continue
                visited.add(nxt)
                parent[nxt] = current
                queue.append(nxt)
                        
        return []
    
class GBFSSearch(BaseSearch):
    def __init__(self, nodes: Dict[str, List[int]], edges: Dict[str, Dict[str, int]], origin: str, destinations: List[str]) -> None:
        super().__init__(nodes, edges, origin, destinations)
        
    def euclidean(self, s1: str, s2: str) -> float:
        """euclidean _summary_
        
        GBFS and A* require what is known as a heurestic function. Since the points are in 2D space, 
        I use the Euclidean Distance, also called the L^2 Norm. The reason is that there is
        a function in the standard library that implement it.

        Args:
            s1 (str): Index of first node
            s2 (str): Index of second node

        Returns:
            float: 
        """
        c1 = self.nodes[s1]
        c2 = self.nodes[s2]
        differences = [x - y for x, y in zip(c1, c2)]
        return hypot(*differences)
    
    def search_with_specific_goal(self, goal: str) -> List[str]:
        """search_with_specific_goal _summary_
        
        Test the GBFS algorithm with a specific goal, note that origin remain the same

        Args:
            goal (str): Goal to test

        Returns:
            SearchResult: _description_
        """
        assert goal in self.nodes.keys(), "Invalid goal"
        prioirty_queue = PriorityQueue[Tuple[float, str]]()
        prioirty_queue.push((self.euclidean(self.origin, goal), self.origin))
        parent: Dict[str, Optional[str]] = {self.origin: None}
        visited: Set[str] = {self.origin}
        
        while len(prioirty_queue) > 0:
            # we don't really need the cost, it's mainly for navigation only
            current = prioirty_queue.pop()[1]
            if current == goal:
                # the usual traceback
                path: List[str] = []
                cur: Optional[str] = current
                while cur is not None:
                    # traceback
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
                
            for node in self.edges.get(current, {}).keys():
                if node in visited: 
                    continue
                visited.add(node)
                parent[node] = current
                prioirty_queue.push((self.euclidean(node, goal), node))
        return []
        
    def heuristic(self, node: str) -> float:
        return min([self.euclidean(node, destination) for destination in self.destinations])
    
    def search(self) -> List[str]:
        
        """search _summary_
        
        Implement the GBFS algorithm.
        Note: so if we assume to find the optimal point by Euclidean distance to the goal....

        Returns:
            SearchResult: _description_
        """
        prioirty_queue = PriorityQueue[Tuple[float, str]]()
        prioirty_queue.push((self.heuristic(self.origin), self.origin))
        parent: Dict[str, Optional[str]] = {self.origin: None}
        visited: Set[str] = {self.origin}
        
        while len(prioirty_queue) > 0:
            current = prioirty_queue.pop()[1]
            if current in self.destinations:
                # the usual traceback
                path: List[str] = []
                cur: Optional[str] = current
                while cur is not None:
                    # traceback
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
                
            for node in self.edges.get(current, {}).keys():
                if node in visited: 
                    continue
                visited.add(node)
                parent[node] = current
                prioirty_queue.push((self.heuristic(node), node))
        return []
    

# so I can steal the heurestic function
class AStarSearch(GBFSSearch):
    def __init__(self, nodes: Dict[str, List[int]], edges: Dict[str, Dict[str, int]], origin: str, destinations: List[str]) -> None:
        super().__init__(nodes, edges, origin, destinations)
        
    def heuristic(self, node: str) -> float:
        h = super().heuristic(node)
        return h 
        
    # def search(self) -> List[str]:
    #     """search _summary_
        
    #     Implement the A* algorithm.
    #     Note: so if we assume to find the optimal point by Euclidean distance to the goal....

    #     Returns:
    #         SearchResult: _description_
    #     """
    #     prioirty_queue = PriorityQueue[Tuple[float, str]]()
    #     prioirty_queue.push((self.heuristic(self.origin), self.origin))
    #     parent: Dict[str, Optional[str]] = {self.origin: None}
        
    #     costs: Dict[str, float] = {node: float("inf") for node in self.nodes}
    #     costs[self.origin] = 0
        
    #     score: Dict[str, float] = {node: float("inf") for node in self.nodes}
    #     score[self.origin] = self.heuristic(self.origin)
        
    #     visited: Set[str] = {self.origin}
        
    #     while len(prioirty_queue) > 0:
    #         current = prioirty_queue.pop()[1]
    #         if current in self.destinations:
    #             # the usual traceback
    #             path: List[str] = []
    #             cur: Optional[str] = current
    #             while cur is not None:
    #                 # traceback
    #                 path.append(cur)
    #                 cur = parent[cur]
    #             path.reverse()
    #             return path
                
    #         for node in self.edges.get(current, {}).keys():
    #             if node in visited: 
    #                 continue
    #             visited.add(node)
    #             costs[node] += self.edges[current][node]
    #             parent[node] = current
    #             prioirty_queue.push((costs[node] + self.heuristic(node), node))
    #     return []

    def search(self) -> List[str]:

        priority_queue = PriorityQueue[Tuple[float, str]]()
        priority_queue.push((self.heuristic(self.origin), self.origin))
        
        parent: Dict[str, Optional[str]] = {self.origin: None}
        
        # g_score
        costs: Dict[str, float] = {node: float("inf") for node in self.nodes}
        costs[self.origin] = 0
        
        while len(priority_queue) > 0:

            current = priority_queue.pop()[1]
            
            if current in self.destinations:

                path = []
                cur = current
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
                
            for node, weight in self.edges.get(current, {}).items():
                # NEW COST: g(current) + edge's weight
                new_g_score = costs[current] + weight
                
                # IF (cheaper path found)
                if new_g_score < costs[node]:
                    parent[node] = current
                    costs[node] = new_g_score
                    # f_score = new_g_score + h(node)
                    f_score = new_g_score + self.heuristic(node)
                    priority_queue.push((f_score, node))
                    
        return []