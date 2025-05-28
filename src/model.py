from src.identification import select_backdoor
from src.estimation import PropensityScoreWeighting
from src.refutation import add_random_confounder, subset_refuter

class CausalModel:
    def __init__(self, data, dag, treatment, outcome):
        self.data, self.dag = data, dag
        self.treatment, self.outcome = treatment, outcome

    def identify_effect(self):
        self.adjustment_set = select_backdoor(self.dag, self.treatment, self.outcome)
        return self.adjustment_set

    def estimate_effect(self, method='ps_weighting', **kw):
        X = self.data[self.adjustment_set]
        w = self.data[self.treatment].astype(int).values
        y = self.data[self.outcome].astype(int).values
        if method=='ps_weighting':
            est = PropensityScoreWeighting()
        self.ate_, self.ate_ci_ = est.estimate(X,w,y,**kw)
        return self.ate_, self.ate_ci_

    def refute(self, method='random_common', **kw):
        if method=='random_common':
            return add_random_confounder(self,**kw)
        else:
            return subset_refuter(self,**kw)