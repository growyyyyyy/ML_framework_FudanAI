#!/usr/bin/env python3
"""
测试修复后的因果推断代码
"""

from src.data import load_and_clean, preprocess_ecommerce_data
from src.graph import CausalDAG
from src.model import CausalModel

def main():
    print("=== 测试因果推断代码修复 ===")
    
    # 1. 加载和预处理数据
    print("1. 加载数据...")
    df = load_and_clean("e_commerce/EComm.csv")
    print(f"原始数据shape: {df.shape}")
    
    print("2. 预处理数据...")
    df = preprocess_ecommerce_data(df)
    print("预处理后的关键变量:")
    print(f"  - HighlyDissatisfied: {df['HighlyDissatisfied'].value_counts().to_dict()}")
    print(f"  - Churn: {df['Churn'].value_counts().to_dict()}")
    
    # 2. 创建简化的因果图进行测试
    print("3. 创建因果图...")
    causal_graph = """digraph G {
      HighlyDissatisfied [label="Highly Dissatisfied", color="green", style="filled"];
      Churn              [label="Churning out",         color="red",   style="filled"];
      Complain           [label="Complaint or not",     color="green", style="filled"];
      Tenure             [label="Tenure with E Commerce"];
      CityTier           [label="City Tier"];
      U [label="Unobserved Confounders", observed="false"];

      U -> HighlyDissatisfied;
      U -> Tenure;
      U -> CityTier;

      HighlyDissatisfied -> Churn;
      Complain -> Churn;
      Complain -> HighlyDissatisfied;
      Tenure -> HighlyDissatisfied;
      CityTier -> Tenure;
    }"""
    
    dag = CausalDAG.from_dot(causal_graph)
    print(f"图节点: {list(dag.g.nodes())}")
    print(f"图边数: {len(dag.g.edges())}")
    
    # 3. 创建因果模型
    print("4. 创建因果模型...")
    cm = CausalModel(df, dag, treatment="HighlyDissatisfied", outcome="Churn")
    
    # 4. 识别因果效应
    print("5. 识别后门调整集...")
    adj = cm.identify_effect()
    print(f"识别的调整集: {adj}")
    
    # 5. 估计因果效应
    print("6. 估计因果效应...")
    try:
        ate, ci = cm.estimate_effect()
        print(f"平均处理效应 (ATE): {ate:.4f}")
        print(f"95%置信区间: [{ci[0]:.4f}, {ci[1]:.4f}]")
        
        # 6. 进行稳健性检查
        print("7. 稳健性检查...")
        try:
            random_refute = cm.refute('random_common')
            print(f"随机混杂因子检验: {random_refute}")
        except Exception as e:
            print(f"随机混杂因子检验失败: {e}")
        
        try:
            subset_refute = cm.refute('subset', frac=0.8)
            print(f"子集检验: {subset_refute}")
        except Exception as e:
            print(f"子集检验失败: {e}")
            
        print("\n=== 测试成功完成！ ===")
        
    except Exception as e:
        print(f"估计因果效应时出错: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main() 