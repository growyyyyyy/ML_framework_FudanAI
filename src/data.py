import pandas as pd


def load_and_clean(path: str):
    df = pd.read_csv(path)
    return df


def preprocess_ecommerce_data(df):
    """
    预处理电商数据，创建因果图中需要的变量
    """
    df = df.copy()
    
    # 创建 HighlyDissatisfied 变量（基于 SatisfactionScore）
    # 假设满意度评分低于等于2为高度不满意
    if 'SatisfactionScore' in df.columns:
        df['HighlyDissatisfied'] = (df['SatisfactionScore'] <= 2).astype(int)
    
    # 确保 Churn 是二进制变量
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].astype(int)
    
    # 确保其他数值变量是适当的类型
    numeric_cols = ['Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp', 
                   'NumberOfDeviceRegistered', 'NumberOfAddress', 'OrderAmountHikeFromlastYear',
                   'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 处理分类变量编码
    categorical_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 
                       'PreferedOrderCat', 'MaritalStatus']
    
    for col in categorical_cols:
        if col in df.columns:
            # 简单的标签编码
            df[col] = pd.Categorical(df[col]).codes
    
    # 处理缺失值
    df = df.fillna(df.median(numeric_only=True))
    
    return df
