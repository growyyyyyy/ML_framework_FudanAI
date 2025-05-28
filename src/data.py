import pandas as pd


def load_and_clean(path: str):
    df = pd.read_csv(path)
    return df
