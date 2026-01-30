from typing import Tuple, List, Dict, Any

def parse_tuple(s: str) -> Tuple[str, ...]:
    """parse_tuple _summary_
    Args:
        tuple_string (str): String to parse as tuple object, formatted as (a1, a2, ..., an)

	Returns:
		Tuple[str, ...]: Parsed tuple
	""" 
    return tuple(str(i) for i in s.replace("(", "").replace(")", "").split(","))

def read_input(filename: str) -> Tuple[
        Dict[str, List[int]], # node with index
        Dict[str, Dict[str, int]], # adjecency edge
        str, # origin
        List[str] # destinations
    ]:
    """read_input _summary_
    
    Reads input for the problem. 
    Note that inputs are read as strings.

	Args:
		filename (str): File that contains input for the problem

	Returns:
		Any: A tuple 
	"""    
    
    l: List[List[str]] = []
    
    nodes: Dict[str, List[int]] = {}
    edges: Dict[str, Dict[str, int]] = {}
    origin = ""
    destinations = []
    with open(filename, "r") as file:
        for line in file.readlines():
            i = line.strip()
            if i in ["Nodes:", "Edges:", "Origin:", "Destinations:"]:
                l.append([])
            else:
                l[-1].append(i.strip("\n"))
    # print(l)
    
    # nodes
    for i in l[0]:
        node, coordinates = i.split(":")
        nodes[node] = [int(i) for i in  parse_tuple(coordinates)]
        
    # edges
    for i in l[1]:
        edge, cost = i.split(":")
        edge_start, edge_end = parse_tuple(edge)
        if edges.get(edge_start) is None:
            edges[edge_start] = {}
        edges[edge_start][edge_end] = int(cost)
        
    origin = l[2][0]
    destinations = [i.strip() for i in l[3][0].split(";") if len(i.strip()) > 0]
    return nodes, edges, origin, destinations