import numpy as np
from sklearn.linear_model import LogisticRegression

class PropensityScoreWeighting:
    def estimate(self, X, w, y, bootstrap_rounds=200, alpha=0.05):
        # 1. 拟合倾向模型
        ps = LogisticRegression(solver='liblinear').fit(X, w).predict_proba(X)[:,1]
        # 2. 计算加权 ATE
        wt_t = w / ps
        wt_c = (1-w) / (1-ps)
        ate = np.average(y[w==1], weights=wt_t[w==1]) - np.average(y[w==0], weights=wt_c[w==0])
        # 3. Bootstrap CI
        boots = []
        n = len(y)
        for _ in range(bootstrap_rounds):
            idx = np.random.choice(n, n, replace=True)
            ps_i, w_i, y_i = ps[idx], w[idx], y[idx]
            wt_ti = w_i / ps_i; wt_ci = (1-w_i) / (1-ps_i)
            boots.append(np.average(y_i[w_i==1],weights=wt_ti[w_i==1])
                         - np.average(y_i[w_i==0],weights=wt_ci[w_i==0]))
        ci = (np.percentile(boots,100*alpha/2), np.percentile(boots,100*(1-alpha/2)))
        return ate, ci