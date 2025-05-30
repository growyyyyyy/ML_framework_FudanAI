from src.identification import select_backdoor
from src.estimation import PropensityScoreWeighting
from src.refutation import add_random_confounder, subset_refuter

class CausalModel:
    def __init__(self, data, dag, treatment, outcome, observed_vars=None):
        self.data, self.dag = data, dag
        self.treatment, self.outcome = treatment, outcome
        
        # 如果没有提供观测变量列表，使用数据集的列名
        if observed_vars is None:
            self.observed_vars = set(self.data.columns)
        else:
            self.observed_vars = set(observed_vars)

    def identify_effect(self):
        self.adjustment_set = select_backdoor(
            self.dag, self.treatment, self.outcome, self.observed_vars
        )
        return self.adjustment_set

    def estimate_effect(self, method='ps_weighting', **kw):
        # 过滤调整集，只保留数据中实际存在的变量
        valid_adjustment_set = [var for var in self.adjustment_set if var in self.data.columns]
        
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
        
        if method=='ps_weighting':
            est = PropensityScoreWeighting()
        self.ate_, self.ate_ci_ = est.estimate(X, w, y, **kw)
        return self.ate_, self.ate_ci_

    def refute(self, method='random_common', **kw):
        if method=='random_common':
            return add_random_confounder(self,**kw)
        else:
            return subset_refuter(self,**kw)