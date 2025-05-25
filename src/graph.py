import networkx as nx
from networkx.drawing.nx_agraph import to_agraph


class CausalDAG:
    def __init__(self, dot_str: str):
        self.dot = dot_str
        self.g = nx.DiGraph(
            nx.drawing.nx_agraph.from_agraph(to_agraph(dot_str)))

    @classmethod
    def from_dot(cls, dot_str: str):
        return cls(dot_str)

    def to_png(self, out_path="causal_model.png", size=(16, 14)):
        A = to_agraph(self.dot)
        A.graph_attr.update(size="{},{}".format(*size))
        A.layout('dot')
        A.draw(out_path)
        return out_path
