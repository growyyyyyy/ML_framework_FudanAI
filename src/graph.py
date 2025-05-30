# src/graph.py

import pydot
import networkx as nx

class CausalDAG:
    def __init__(self, dot_str: str):
        self.dot = dot_str
        # 解析DOT字符串并创建NetworkX图
        self._parse_dot_to_networkx()

    def _parse_dot_to_networkx(self):
        """将DOT字符串解析为NetworkX有向图"""
        # 使用pydot解析DOT字符串
        graphs = pydot.graph_from_dot_data(self.dot)
        if not graphs:
            raise ValueError("无法解析 DOT 数据")
        
        pydot_graph = graphs[0]
        
        # 创建NetworkX有向图
        self.g = nx.DiGraph()
        
        # 添加节点
        for node in pydot_graph.get_nodes():
            node_name = node.get_name().strip('"')
            if node_name not in ['node', 'edge', 'graph']:  # 跳过默认属性节点
                self.g.add_node(node_name)
        
        # 添加边
        for edge in pydot_graph.get_edges():
            source = edge.get_source().strip('"')
            target = edge.get_destination().strip('"')
            self.g.add_edge(source, target)

    @classmethod
    def from_dot(cls, dot_str: str):
        # 创建实例并解析DOT字符串
        return cls(dot_str)

    def to_png(self, path: str):
        # 直接让 pydot 把 dot 渲染为 PNG
        graphs = pydot.graph_from_dot_data(self.dot)
        if not graphs:
            raise ValueError("无法解析 DOT 数据")
        graphs[0].write_png(path)