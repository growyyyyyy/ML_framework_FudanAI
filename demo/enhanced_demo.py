#!/usr/bin/env python3
"""
增强版因果推断框架演示
展示各种机器学习方法在因果推断中的应用
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import load_and_clean, preprocess_ecommerce_data
from src.graph import CausalDAG
from src.model import CausalModel

def main():
    print("=== 增强版因果推断框架演示 ===\n")
    
    # 1. 数据准备
    print("1. 数据准备...")
    df_raw = load_and_clean("demo/e_commerce/EComm.csv")
    df = preprocess_ecommerce_data(df_raw)
    print(f"数据shape: {df.shape}")
    print(f"HighlyDissatisfied分布: {df['HighlyDissatisfied'].value_counts().to_dict()}")
    print(f"Churn分布: {df['Churn'].value_counts().to_dict()}\n")
    
    # 2. 因果图构建
    print("2. 构建因果图...")
    causal_graph = """digraph G {
      HighlyDissatisfied [label="Highly Dissatisfied", color="green", style="filled"];
      Churn              [label="Churning out",         color="red",   style="filled"];
      Complain           [label="Complaint or not",     color="green", style="filled"];
      Tenure             [label="Tenure with E Commerce"];
      CityTier           [label="City Tier"];
      OrderCount         [label="Order Count"];
      CouponUsed         [label="Coupon Used"];
      U [label="Unobserved Confounders", observed="false"];

      U -> HighlyDissatisfied;
      U -> Tenure;
      U -> CityTier;

      HighlyDissatisfied -> Churn;
      Complain -> Churn;
      Complain -> HighlyDissatisfied;
      Tenure -> HighlyDissatisfied;
      CityTier -> OrderCount;
      OrderCount -> Churn;
      CouponUsed -> OrderCount;
    }"""
    
    dag = CausalDAG.from_dot(causal_graph)
    cm = CausalModel(df, dag, treatment="HighlyDissatisfied", outcome="Churn")
    
    # 3. 后门调整集识别
    print("3. 识别后门调整集...")
    adj = cm.identify_effect()
    print(f"识别的调整集: {adj}\n")
    
    # 4. 方法比较
    print("4. 比较不同的机器学习估计方法...")
    print("-" * 50)
    
    methods_to_compare = [
        'ps_weighting',
        'improved_ps_weighting', 
        's_learner',
        't_learner',
        'x_learner',
        # 'double_ml',
        # 'causal_forest'
    ]
    
    results = cm.compare_methods(
        methods=methods_to_compare,
        bootstrap_rounds=50,  # 减少bootstrap轮数以加快演示
        alpha=0.05
    )
    print()
    
    # 5. 详细分析最佳方法
    print("5. 详细分析改进的倾向评分方法...")
    print("-" * 50)
    
    # 使用自动模型选择
    print("正在进行改进的倾向评分估计...")
    ate_improved, ci_improved = cm.estimate_effect(
        method='improved_ps_weighting',
        auto_select_model=True,
        trim_threshold=0.05,
        bootstrap_rounds=50  # 减少bootstrap轮数以加快演示
    )
    
    print(f"改进倾向评分 - ATE: {ate_improved:.4f}, CI: [{ci_improved[0]:.4f}, {ci_improved[1]:.4f}]")
    
    # 诊断信息
    diagnostics = cm.diagnostics(method='improved_ps_weighting', auto_select_model=True)
    print("\n倾向评分诊断:")
    print(f"  处理组倾向评分均值: {diagnostics['propensity_score_summary']['treated']['mean']:.3f}")
    print(f"  对照组倾向评分均值: {diagnostics['propensity_score_summary']['control']['mean']:.3f}")
    print(f"  重叠区间比例: {diagnostics['overlap']['proportion_in_overlap']:.3f}")
    print(f"  大权重比例: {diagnostics['weight_summary']['proportion_large_weights']:.3f}")
    print()
    
    # 6. 异质性处理效应分析
    print("6. 异质性处理效应分析...")
    print("-" * 50)
    
    try:
        # 使用因果森林估计个体效应
        individual_effects_cf = cm.estimate_heterogeneous_effects(
            method='causal_forest',
            n_estimators=50,  # 减少树数量以加快演示
            bootstrap_rounds=25  # 减少bootstrap轮数以加快演示
        )
        
        print(f"因果森林 - 个体效应统计:")
        print(f"  均值: {np.mean(individual_effects_cf):.4f}")
        print(f"  标准差: {np.std(individual_effects_cf):.4f}")
        print(f"  最小值: {np.min(individual_effects_cf):.4f}")
        print(f"  最大值: {np.max(individual_effects_cf):.4f}")
        
        # 使用T-Learner估计个体效应
        individual_effects_t = cm.estimate_heterogeneous_effects(
            method='t_learner',
            base_model='gradient_boosting'
        )
        
        print(f"\nT-Learner - 个体效应统计:")
        print(f"  均值: {np.mean(individual_effects_t):.4f}")
        print(f"  标准差: {np.std(individual_effects_t):.4f}")
        print(f"  最小值: {np.min(individual_effects_t):.4f}")
        print(f"  最大值: {np.max(individual_effects_t):.4f}")
        
    except Exception as e:
        print(f"异质性效应估计失败: {e}")
    
    print()
    
    # 7. 双重机器学习分析
    print("7. 双重机器学习分析...")
    print("-" * 50)
    
    try:
        ate_dml, ci_dml = cm.estimate_effect(
            method='double_ml',
            outcome_model='gradient_boosting',
            propensity_model='gradient_boosting',
            n_folds=3,  # 减少折数以加快演示
            bootstrap_rounds=25  # 减少bootstrap轮数以加快演示
        )
        
        print(f"双重ML - ATE: {ate_dml:.4f}, CI: [{ci_dml[0]:.4f}, {ci_dml[1]:.4f}]")
        
        # 获取残差诊断
        residuals = cm.estimator_.get_residuals(
            cm.data[[var for var in adj if var in cm.data.columns]].values,
            cm.data[cm.treatment].astype(int).values,
            cm.data[cm.outcome].astype(int).values
        )
        
        print(f"\n双重ML残差诊断:")
        print(f"  Y残差均值: {np.mean(residuals['y_residuals']):.4f}")
        print(f"  W残差均值: {np.mean(residuals['w_residuals']):.4f}")
        print(f"  Y残差标准差: {np.std(residuals['y_residuals']):.4f}")
        print(f"  W残差标准差: {np.std(residuals['w_residuals']):.4f}")
        
    except Exception as e:
        print(f"双重ML估计失败: {e}")
    
    print()
    
    # 8. 可视化结果
    print("8. 生成可视化结果...")
    print("-" * 50)
    
    # 收集有效结果
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        # 创建结果比较图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ATE比较
        methods = list(valid_results.keys())
        ates = [valid_results[m]['ate'] for m in methods]
        cis_lower = [valid_results[m]['ci'][0] for m in methods]
        cis_upper = [valid_results[m]['ci'][1] for m in methods]
        ci_widths = [valid_results[m]['ci_width'] for m in methods]
        
        # 计算误差条长度，确保为非负值
        lower_errors = np.abs(np.array(ates) - np.array(cis_lower))
        upper_errors = np.abs(np.array(cis_upper) - np.array(ates))
        
        ax1.errorbar(range(len(methods)), ates, 
                    yerr=[lower_errors, upper_errors],
                    fmt='o', capsize=5, capthick=2)
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        ax1.set_ylabel('Average Treatment Effect')
        ax1.set_title('ATE Estimates by Method')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 置信区间宽度比较
        ax2.bar(range(len(methods)), ci_widths, alpha=0.7)
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        ax2.set_ylabel('Confidence Interval Width')
        ax2.set_title('Precision Comparison (Smaller is Better)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('causal_methods_comparison.png', dpi=300, bbox_inches='tight')
        print("方法比较图已保存为 'causal_methods_comparison.png'")
        
        # 异质性效应分布图
        if 'individual_effects_cf' in locals():
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.hist(individual_effects_cf, bins=30, alpha=0.7, density=True, 
                    color='skyblue', edgecolor='black')
            plt.axvline(np.mean(individual_effects_cf), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(individual_effects_cf):.3f}')
            plt.xlabel('Individual Treatment Effect')
            plt.ylabel('Density')
            plt.title('Distribution of Individual Effects (Causal Forest)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.hist(individual_effects_t, bins=30, alpha=0.7, density=True, 
                    color='lightgreen', edgecolor='black')
            plt.axvline(np.mean(individual_effects_t), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(individual_effects_t):.3f}')
            plt.xlabel('Individual Treatment Effect')
            plt.ylabel('Density')
            plt.title('Distribution of Individual Effects (T-Learner)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('heterogeneous_effects.png', dpi=300, bbox_inches='tight')
            print("异质性效应分布图已保存为 'heterogeneous_effects.png'")
        
        plt.show()
    
    # 9. 稳健性检验
    print("\n9. 稳健性检验...")
    print("-" * 50)
    
    try:
        random_refute = cm.refute('random_common')
        subset_refute = cm.refute('subset', frac=0.8)
        
        print(f"随机混杂因子检验: 原始 = {random_refute['orig']:.4f}, 扰动后 = {random_refute['new']:.4f}")
        print(f"子集检验: 原始 = {subset_refute['orig']:.4f}, 子集 = {subset_refute['new']:.4f}")
    except Exception as e:
        print(f"稳健性检验失败: {e}")
    
    # 10. 总结
    print("\n" + "="*60)
    print("总结报告")
    print("="*60)
    
    if valid_results:
        best_method = min(valid_results.items(), key=lambda x: x[1]['ci_width'])
        print(f"最精确的方法: {best_method[0]} (置信区间宽度: {best_method[1]['ci_width']:.4f})")
        print(f"该方法的ATE估计: {best_method[1]['ate']:.4f}")
        print(f"95%置信区间: [{best_method[1]['ci'][0]:.4f}, {best_method[1]['ci'][1]:.4f}]")
        
        # 一致性检查
        all_ates = [v['ate'] for v in valid_results.values()]
        ate_std = np.std(all_ates)
        print(f"\n方法间ATE估计的标准差: {ate_std:.4f}")
        if ate_std < 0.01:
            print("✓ 不同方法的估计结果高度一致")
        elif ate_std < 0.05:
            print("? 不同方法的估计结果较为一致")
        else:
            print("⚠ 不同方法的估计结果存在较大差异，需要进一步调查")
    
    print(f"\n解释: 高度不满意对客户流失的因果效应为负值，")
    print(f"这可能意味着表达不满的客户反而不太容易流失，")
    print(f"可能是因为他们的问题得到了关注和解决。")
    
    print("\n演示完成！")

if __name__ == "__main__":
    main() 