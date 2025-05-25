import pandas as pd


def load_and_clean(path: str):
    df = pd.read_csv(path)
    # 衍生处理变量
    df['HighlyDissatisfied'] = df['SatisfactionScore'] <= 1
    # 删除无用列 & 丢弃缺失
    df = df.drop(['CustomerID', 'SatisfactionScore'], axis=1).dropna()
    return df
