import numpy as np


def add_random_confounder(model, **kw):
    df = model.data.copy()
    df['U_rand'] = np.random.randn(len(df))
    m2 = model.__class__(df, model.dag, model.treatment, model.outcome)
    m2.adjustment_set = model.adjustment_set + ['U_rand']
    ate2, _ = m2.estimate_effect(**kw)
    return {'orig': model.ate_, 'new': ate2}


def subset_refuter(model, frac=0.8, **kw):
    df = model.data.sample(frac=frac, random_state=1)
    m2 = model.__class__(df, model.dag, model.treatment, model.outcome)
    m2.adjustment_set = model.adjustment_set
    ate2, _ = m2.estimate_effect(**kw)
    return {'orig': model.ate_, 'new': ate2}
