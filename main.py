from input import read_input
from search import DFSSearch, BFSSearch, SearchResult, GBFSSearch, AStarSearch
from typing import Optional
import argparse
    
if __name__ == "__main__":
    nodes, edges, origin, destinations = read_input("./example.txt")
    
    # CLI arguments
    parser = argparse.ArgumentParser(
        prog="Route Finder", description="COS30019 Route Finding Program for Assignment 2A"
    )
    
    parser.add_argument("-a", "--algo",  
                        choices=["dfs", "bfs", "gbfs", "astar"], 
                        default="dfs", nargs=1, help="Algorithm to choose (DFS, BFS, GBFS, A*)")
    
    args = parser.parse_args()
    print("Parsed args:", args)
    
    algo = str(args.algo[0]).lower()
    print(algo)
    
    result: Optional[SearchResult] = None
    if algo == "dfs":
        result = DFSSearch(nodes, edges, origin, destinations).search()
    elif algo == "bfs":
        result = BFSSearch(nodes, edges, origin, destinations).search()
    elif algo == "gbfs":
        result = GBFSSearch(nodes, edges, origin, destinations).search()
    elif algo == "astar":
        result = AStarSearch(nodes, edges, origin, destinations).search()
    
    result = AStarSearch(nodes, edges, origin, destinations).search()
    assert result is not None, "yeah you screw up"
    print("Nodes:", nodes)
    print("Edges:", edges)
    print("Origin:", origin)
    print("Destinations:", destinations)
    print("Path:" + "->".join(result.path))
    print(f"Total cost: {result.cost()}")