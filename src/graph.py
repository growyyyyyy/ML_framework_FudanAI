# src/graph.py

import pydot

class CausalDAG:
    def __init__(self, dot_str: str):
        self.dot = dot_str

    @classmethod
    def from_dot(cls, dot_str: str):
        # 只保存字符串，不转换成 networkx
        return cls(dot_str)

    def to_png(self, path: str):
        # 直接让 pydot 把 dot 渲染为 PNG
        graphs = pydot.graph_from_dot_data(self.dot)
        if not graphs:
            raise ValueError("无法解析 DOT 数据")
        graphs[0].write_png(path)