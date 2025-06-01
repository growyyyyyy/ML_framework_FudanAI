from src.identification import select_backdoor
from src.refutation import add_random_confounder, subset_refuter

# 导入所有新的估计器
from src.modern_estimators.propensity_score import PropensityScoreWeighting, ImprovedPropensityScoreWeighting
from src.modern_estimators.meta_learners import SLearner, TLearner, XLearner, RLearner
from src.modern_estimators.double_ml import DoubleMLEstimator
from src.modern_estimators.causal_forest import CausalForestEstimator

class CausalModel:
    def __init__(self, data, dag, treatment, outcome, observed_vars=None):
        self.data, self.dag = data, dag
        self.treatment, self.outcome = treatment, outcome
        
        # 如果没有提供观测变量列表，使用数据集的列名
        if observed_vars is None:
            self.observed_vars = set(self.data.columns)
        else:
            self.observed_vars = set(observed_vars)
        
        # 支持的估计方法
        self.available_methods = {
            'ps_weighting': PropensityScoreWeighting,
            'improved_ps_weighting': ImprovedPropensityScoreWeighting,
            's_learner': SLearner,
            't_learner': TLearner,
            'x_learner': XLearner,
            'r_learner': RLearner,
            'double_ml': DoubleMLEstimator,
            'causal_forest': CausalForestEstimator
        }

    def identify_effect(self):
        self.adjustment_set = select_backdoor(
            self.dag, self.treatment, self.outcome, self.observed_vars
        )
        return self.adjustment_set

    def estimate_effect(self, method='ps_weighting', **kw):
        """
        估计因果效应
        
        Parameters:
        -----------
        method : str
            估计方法，可选：
            - 'ps_weighting': 原始倾向评分加权
            - 'improved_ps_weighting': 改进的倾向评分加权
            - 's_learner': S-Learner元学习器
            - 't_learner': T-Learner元学习器
            - 'x_learner': X-Learner元学习器
            - 'r_learner': R-Learner元学习器
            - 'double_ml': 双重机器学习
            - 'causal_forest': 因果森林
        **kw : dict
            传递给估计器的额外参数
        """
        # 分离构造函数参数和估计参数
        estimate_params = {
            'bootstrap_rounds': kw.pop('bootstrap_rounds', 200),
            'alpha': kw.pop('alpha', 0.05)
        }
        
        # 过滤调整集，只保留数据中实际存在的变量
        if hasattr(self, 'adjustment_set'):
            valid_adjustment_set = [var for var in self.adjustment_set if var in self.data.columns]
        else:
            valid_adjustment_set = []
        
        if not valid_adjustment_set:
            # 如果没有有效的调整变量，使用空的DataFrame
            import pandas as pd
            X = pd.DataFrame(index=self.data.index)
        else:
            X = self.data[valid_adjustment_set]
        
        # 检查处理变量和结果变量是否存在
        if self.treatment not in self.data.columns:
            raise ValueError(f"处理变量 '{self.treatment}' 不存在于数据中。可用变量: {list(self.data.columns)}")
        if self.outcome not in self.data.columns:
            raise ValueError(f"结果变量 '{self.outcome}' 不存在于数据中。可用变量: {list(self.data.columns)}")
            
        w = self.data[self.treatment].astype(int).values
        y = self.data[self.outcome].astype(int).values
        
        # 选择估计器
        if method not in self.available_methods:
            raise ValueError(f"不支持的方法: {method}. 可用方法: {list(self.available_methods.keys())}")
        
        estimator_class = self.available_methods[method]
        
        # 处理不同估计器的参数（只传递构造函数参数）
        if method == 'ps_weighting':
            # 原始方法，保持向后兼容
            estimator = estimator_class()
        else:
            # 其他方法，传递构造函数参数
            estimator = estimator_class(**kw)
        
        # 估计效应（传递估计参数）
        self.ate_, self.ate_ci_ = estimator.estimate(X, w, y, **estimate_params)
        self.estimator_ = estimator
        
        return self.ate_, self.ate_ci_
    
    def estimate_heterogeneous_effects(self, method='causal_forest', **kw):
        """
        估计异质性处理效应 (Heterogeneous Treatment Effects)
        
        Parameters:
        -----------
        method : str
            支持异质性效应估计的方法
        **kw : dict
            传递给估计器的额外参数
            
        Returns:
        --------
        individual_effects : array-like
            每个个体的处理效应估计
        """
        # 分离估计参数和构造函数参数
        estimate_params = {
            'bootstrap_rounds': kw.pop('bootstrap_rounds', 200),
            'alpha': kw.pop('alpha', 0.05)
        }
        
        # 准备数据
        valid_adjustment_set = [var for var in getattr(self, 'adjustment_set', []) if var in self.data.columns]
        
        if not valid_adjustment_set:
            import pandas as pd
            X = pd.DataFrame(index=self.data.index)
        else:
            X = self.data[valid_adjustment_set]
        
        w = self.data[self.treatment].astype(int).values
        y = self.data[self.outcome].astype(int).values
        
        # 选择支持异质性效应的估计器
        heterogeneous_methods = ['s_learner', 't_learner', 'x_learner', 'causal_forest']
        
        if method not in heterogeneous_methods:
            raise ValueError(f"方法 '{method}' 不支持异质性效应估计。支持的方法: {heterogeneous_methods}")
        
        estimator_class = self.available_methods[method]
        # 只传递构造函数参数给估计器
        estimator = estimator_class(**kw)
        
        # 估计条件处理效应
        individual_effects = estimator.estimate_conditional_effects(X, w, y)
        
        return individual_effects
    
    def compare_methods(self, methods=None, **kw):
        """
        比较多种估计方法
        
        Parameters:
        -----------
        methods : list or None
            要比较的方法列表，如果为None则使用所有方法
        **kw : dict
            传递给估计器的额外参数
            
        Returns:
        --------
        results : dict
            每种方法的估计结果
        """
        if methods is None:
            methods = ['ps_weighting', 'improved_ps_weighting', 's_learner', 
                      't_learner', 'double_ml', 'causal_forest']
        
        results = {}
        
        for method in methods:
            try:
                print(f"正在估计: {method}")
                ate, ci = self.estimate_effect(method=method, **kw)
                results[method] = {
                    'ate': ate,
                    'ci': ci,
                    'ci_width': ci[1] - ci[0]
                }
                print(f"{method}: ATE = {ate:.4f}, CI = [{ci[0]:.4f}, {ci[1]:.4f}]")
            except Exception as e:
                print(f"{method} failed: {e}")
                results[method] = {'error': str(e)}
        
        return results
    
    def diagnostics(self, method='improved_ps_weighting', **kw):
        """
        运行诊断检查
        
        Parameters:
        -----------
        method : str
            要进行诊断的方法
        **kw : dict
            传递给估计器的额外参数
            
        Returns:
        --------
        diagnostics : dict
            诊断结果
        """
        # 先运行估计
        self.estimate_effect(method=method, **kw)
        
        # 获取诊断信息
        if hasattr(self.estimator_, 'diagnostics'):
            valid_adjustment_set = [var for var in getattr(self, 'adjustment_set', []) if var in self.data.columns]
            
            if not valid_adjustment_set:
                import pandas as pd
                X = pd.DataFrame(index=self.data.index)
            else:
                X = self.data[valid_adjustment_set]
            
            w = self.data[self.treatment].astype(int).values
            
            return self.estimator_.diagnostics(X, w)
        else:
            return {"message": f"方法 '{method}' 不支持诊断功能"}

    def refute(self, method='random_common', **kw):
        if method=='random_common':
            return add_random_confounder(self,**kw)
        else:
            return subset_refuter(self,**kw)