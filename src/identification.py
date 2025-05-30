import networkx as nx

from src.graph import CausalDAG

def select_backdoor(dag: CausalDAG, treat: str, outcome: str, observed_vars=None):
    """
    选择后门调整集
    Args:
        dag: 因果DAG
        treat: 处理变量
        outcome: 结果变量
        observed_vars: 观测到的变量集合，如果为None则从DAG中的所有节点中排除标记为未观测的节点
    """
    G = dag.g.to_undirected()
    
    # 如果没有提供观测变量列表，则假设除了特殊标记的节点外都是观测的
    if observed_vars is None:
        # 默认排除名为 'U' 的节点（通常表示未观测混杂因子）
        observed_vars = set(dag.g.nodes()) - {'U'}
    
    backdoors = set()
    
    # 寻找从treat到outcome的所有路径
    try:
        for path in nx.all_simple_paths(G, source=treat, target=outcome):
            if len(path) > 2:  # 路径长度至少为3（treat-confonder-outcome）
                # 检查第一步是否是反向边（treat <- X）
                if dag.g.has_edge(path[1], treat):
                    # 将路径中间的节点加入后门集候选
                    for node in path[1:-1]:
                        if node in observed_vars:  # 只包含观测到的变量
                            backdoors.add(node)
    except nx.NetworkXNoPath:
        # 如果没有路径，返回空集
        pass
    
    return list(backdoors)