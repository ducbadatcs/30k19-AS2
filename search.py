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
    def search(self) -> Tuple[List[str], int]:
        """search _summary_
        
        Abstract Method for implementing search algorithms.

		Returns:
			SearchModel: _description_
		"""        
        return ([], 0)
    
    # helper methods
    def cost(self, path: List[str]) -> int:
        """cost Helper method to get cost of path

        Args:
            path (List[str]): _description_

        Returns:
            int: _description_
        """
        if len(path) == 0 or self.edges is None : return -1
        
        c = 0
        for start, end in zip(path[:-1], path[1:]):
            c += self.edges[start][end]
        return c
    
    def reconstruct_path(self, current: str, parent: Dict[str, Optional[str]]) -> List[str]:
        """reconstruct_path Helper to reconstruct path to the origin.

        Args:
            current (str): _description_
            parent (Dict[str, Optional[str]]): _description_

        Returns:
            List[str]: _description_
        """
        path = []
        cur = current
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()
        return path
    
        
class DFSSearch(BaseSearch):
    def __init__(self, nodes: Dict[str, List[int]], edges: Dict[str, Dict[str, int]], origin: str, destinations: List[str]) -> None:
        super().__init__(nodes, edges, origin, destinations)
        
    def search(self) -> Tuple[List[str], int]:
        """search _summary_
        
        Implement the DFS algorithm:

		Returns:
			SearchModel: _description_
		""" 
        stack: Deque[str] = deque([self.origin])
        parent: Dict[str, Optional[str]] = {self.origin: None}
        visited: Set[str] = {self.origin}

        nodes_expanded = 0
        
        while len(stack) > 0:
            current = stack.pop()
            nodes_expanded += 1

            if current in self.destinations:
                # reconstruct path
                return (self.reconstruct_path(current, parent), nodes_expanded)
            
            neighbors = sorted(self.edges.get(current, {}).keys(), reverse=True)

            for node in neighbors:
                if node in visited:
                    continue
                visited.add(node)
                parent[node] = current
                stack.append(node)
        return ([], nodes_expanded)

class BFSSearch(BaseSearch):
    def __init__(self, nodes: Dict[str, List[int]], edges: Dict[str, Dict[str, int]], origin: str, destinations: List[str]) -> None:
        super().__init__(nodes, edges, origin, destinations)
        
    def search(self) -> Tuple[List[str], int]:
        """search _summary_
        
        Implement the BFS algorithm:

		Returns:
			SearchModel: _description_
		""" 
        queue: Deque[str] = deque([self.origin])
        parent: Dict[str, Optional[str]] = {self.origin: None}
        visited: Set[str] = {self.origin}

        nodes_expanded = 0
        
        while len(queue) > 0:
            current = queue.popleft()
            nodes_expanded += 1

            # get current point. If it is a destination, return.
            if current in self.destinations:
                return (self.reconstruct_path(current, parent), nodes_expanded)
            
            neighbors = sorted(self.edges.get(current, {}).keys())

            # otherwise, keep searching.
            for nxt in neighbors:
                if nxt in visited:
                    continue
                visited.add(nxt)
                parent[nxt] = current
                queue.append(nxt)
                        
        return [], nodes_expanded
    
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
    
    def search_with_specific_goal(self, goal: str) -> Tuple[List[str], int]:
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
        nodes_expanded = 0
        
        while len(prioirty_queue) > 0:
            
            # we don't really need the cost, it's mainly for navigation only
            current = prioirty_queue.pop()[1]
            nodes_expanded += 1
            if current == goal:
                return (self.reconstruct_path(current, parent), nodes_expanded)
                
            for node in self.edges.get(current, {}).keys():
                if node in visited: 
                    continue
                visited.add(node)
                parent[node] = current
                prioirty_queue.push((self.euclidean(node, goal), node))
        return ([], nodes_expanded)
        
    def heuristic(self, node: str) -> float:
        return min([self.euclidean(node, destination) for destination in self.destinations])
    
    def search(self) -> Tuple[List[str], int]:
        
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

        nodes_expanded = 0
        
        while len(prioirty_queue) > 0:
            current = prioirty_queue.pop()[1]
            nodes_expanded += 1

            if current in self.destinations:
                # the usual traceback
                return (self.reconstruct_path(current, parent), nodes_expanded)
                
            for node in self.edges.get(current, {}).keys():
                if node in visited: 
                    continue
                visited.add(node)
                parent[node] = current
                prioirty_queue.push((self.heuristic(node), node))
        return [], nodes_expanded
    

# so I can steal the heurestic function
class AStarSearch(GBFSSearch):
    def __init__(self, nodes: Dict[str, List[int]], edges: Dict[str, Dict[str, int]], origin: str, destinations: List[str]) -> None:
        super().__init__(nodes, edges, origin, destinations)
        
    def heuristic(self, node: str) -> float:
        h = super().heuristic(node)
        return h 

    def search(self) -> Tuple[List[str], int]:

        priority_queue = PriorityQueue[Tuple[float, str]]()
        priority_queue.push((self.heuristic(self.origin), self.origin))
        
        parent: Dict[str, Optional[str]] = {self.origin: None}
        
        # g_score
        costs: Dict[str, float] = {node: float("inf") for node in self.nodes}
        costs[self.origin] = 0

        nodes_expanded = 0
        
        while len(priority_queue) > 0:

            current = priority_queue.pop()[1]
            nodes_expanded += 1
            
            if current in self.destinations:
                return (self.reconstruct_path(current, parent), nodes_expanded)
                
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
                    
        return [], nodes_expanded
    
class UCSSearch(BaseSearch):

    def __init__(self, nodes: Dict[str, List[int]], edges: Dict[str, Dict[str, int]], origin: str, destinations: List[str]) -> None:
        super().__init__(nodes, edges, origin, destinations)
        
    def search(self) -> Tuple[List[str], int]:
        
        # Difference from A*: priority = new_cost (no heuristic).
        # Uniform Cost Search (UCS) is a search algorithm used in artificial intelligence (AI) for finding the least cost path in a graph. It is a variant of Dijkstra's algorithm and is particularly useful when all edges of the graph have different weights, and the goal is to find the path with the minimum total cost from a start node to a goal node.
    
        # 1. Khởi tạo Frontier
        frontier = PriorityQueue[Tuple[float, str]]() 
        frontier.push((0, self.origin))
        
        # 2. Khởi tạo các Dict lưuh trữ
        came_from: Dict[str, Optional[str]] = {}
        cost_so_far: Dict[str, float] = {}
        
        came_from[self.origin] = None
        cost_so_far[self.origin] = 0
        
        final_destination = None
        nodes_expanded = 0

        while len(frontier) > 0:
            # Lấy phần tử có chi phí tích lũy thấp nhất
            _, current = frontier.pop()
            nodes_expanded += 1

            # Early Exit: Nếu chạm đích thì dừng ngay
            if current in self.destinations:
                return (self.reconstruct_path(current, came_from), nodes_expanded)
                break
            
            # Duyệt các hàng xóm
            for next_node, weight in self.edges.get(current, {}).items():
                new_cost = cost_so_far[current] + weight
                
                # Logic cốt lõi giống A*: Nếu tìm thấy đường rẻ hơn
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    
                    # DIJKSTRA KEY DIFFERENCE:
                    # Độ ưu tiên CHỈ LÀ chi phí thực tế (g), không cộng thêm heuristic.
                    priority = new_cost 
                    
                    frontier.push((priority, next_node))
                    came_from[next_node] = current
        
            
        return ([], nodes_expanded)

class IDAStarSearch(BaseSearch):
    def __init__(self, 
                 nodes: Dict[str, List[int]],
                 edges: Dict[str, Dict[str, int]], 
                 origin: str,
                 destinations: List[str]) -> None:
        super().__init__(nodes, edges, origin, destinations)
        self.node_expanded = 0

    # Heuristic h(n):
    # Ước lượng chi phí thấp nhất từ node hiện tại tới đích gần nhất
    # Ở đây dùng khoảng cách Euclidean giữa các node
    def heuristic(self, node: str) -> float:
        def euclidean(s1: str, s2: str) -> float:
            c1 = self.nodes[s1]
            c2 = self.nodes[s2]
            return hypot(c1[0] - c2[0], c1[1] - c2[1])

        return min(euclidean(node, d) for d in self.destinations)

    def search(self) -> Tuple[List[str], int]:
        # threshold: ngưỡng f-cost ban đầu
        # f(n) = g(n) + h(n)
        self.node_expanded = 0
        threshold = self.heuristic(self.origin)
        # path: danh sách node biểu diễn đường đi hiện tại từ gốc tới node đang xét
        path: List[str] = [self.origin]

        while True:
        # Bắt đầu DFS giới hạn bởi threshold
            temp = self._dfs(path, 0, threshold)
            if temp == "FOUND":
                return path, self.node_expanded
            if temp == float("inf"):
                return [], self.node_expanded
            threshold = temp

    def _dfs(self, path: List[str], g: float, threshold: float):
        #DFS có giới hạn f-cost

        #path[-1] : node hiện tại (current state)
        #g        : chi phí từ start -> node hiện tại (g(n))
        #h(n)     : heuristic(node)
        #f(n)     : g(n) + h(n)

        current = path[-1]
        self.node_expanded += 1

        f = g + self.heuristic(current)

        # Nếu vượt ngưỡng
        if f > threshold:
            return f

        # Nếu là đích
        if current in self.destinations:
            return "FOUND"

        min_threshold = float("inf")

        neighbors = sorted(self.edges.get(current, {}).items())

        for neighbor, cost in neighbors:
            if neighbor in path:
                continue  # tránh chu trình

            path.append(neighbor)
            temp = self._dfs(path, g + cost, threshold)

            if temp == "FOUND":
                return "FOUND"

            min_threshold = min(min_threshold, temp)
            path.pop()

        return min_threshold
