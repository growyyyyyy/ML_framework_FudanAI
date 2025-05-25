import networkx as nx

from src.graph import CausalDAG

def select_backdoor(dag: CausalDAG, treat: str, outcome: str):
    G = dag.g.to_undirected()
    backdoors = set()
    for path in nx.all_simple_paths(G, source=treat, target=outcome):
        # 要求第一步是反向边 treat<-X
        if dag.g.has_edge(path[1], treat):
            backdoors.update(path[1:-1])
    return list(backdoors)