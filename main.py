from input import read_input
from sys import argv
from search import DFSSearch, BFSSearch, GBFSSearch, AStarSearch, BaseSearch, UCSSearch, IDAStarSearch
from typing import Optional, List
import argparse
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Route Finder", description="COS30019 Route Finding Program for Assignment 2A"
    )
    
    if len(argv) != 3:
        raise Exception("Error: Invalid number of arguments.")
    
    filename = argv[1]
    method = argv[2]
    
    nodes, edges, origin, destinations = read_input(filename)
    
    result: Optional[BaseSearch] = None
    
    match method:
        case "dfs": result = DFSSearch(nodes, edges, origin, destinations)
        case "bfs": result = BFSSearch(nodes, edges, origin, destinations)
        case "gbfs": result = GBFSSearch(nodes, edges, origin, destinations)
        case "astar": result = AStarSearch(nodes, edges, origin, destinations)
        case "ucs": result = UCSSearch(nodes, edges, origin, destinations)
        case "idastar": result = IDAStarSearch(nodes, edges, origin, destinations)
        case _:
            raise Exception("Invalid Algorithm")
    
    
    assert result is not None, "Error: Invalid Algorithm"
    
    
    path = result.search()
    
    if len(path) > 0:
        print(f"{filename} {method}")
        print(f"{path[-1]} {len(path)}")
        print("->".join(path))
    else:
        print("Path not found.")
        
    with open("result.log", "w") as res:
        res.writelines([
            "Nodes: " + str(nodes),
            "Edges: " + str(edges),
            "Origin: " + str(origin),
            "Destinations: " + str(destinations),
            "Path:" + "->".join(path),
            "Path cost: " + str(result.cost(path))
        ])