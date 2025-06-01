"""
因果效应估计器基类
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.base import clone


class CausalEstimator(ABC):
    """所有因果效应估计器的基类"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.ate_ = None
        self.ate_ci_ = None
        self.fitted_ = False
    
    @abstractmethod
    def estimate(self, X, w, y, **kwargs):
        """
        估计平均处理效应
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            协变量矩阵
        w : array-like of shape (n_samples,)
            处理变量 (0/1)
        y : array-like of shape (n_samples,)
            结果变量
            
        Returns:
        --------
        ate : float
            平均处理效应
        ci : tuple
            置信区间
        """
        pass
    
    def bootstrap_confidence_interval(self, X, w, y, n_bootstrap=200, alpha=0.05):
        """Bootstrap置信区间"""
        n_samples = len(y)
        bootstrap_ates = []
        failed_count = 0
        
        for i in range(n_bootstrap):
            # 重采样
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
            w_boot = w[indices]
            y_boot = y[indices]
            
            # 估计ATE
            try:
                ate_boot, _ = self._estimate_ate(X_boot, w_boot, y_boot)
                bootstrap_ates.append(ate_boot)
            except Exception as e:
                failed_count += 1
                # 只在前5次失败时显示错误，避免过多输出
                if failed_count <= 5:
                    print(f"Bootstrap round {i+1} failed: {e}")
                continue
        
        # 检查是否有足够的成功bootstrap样本
        if len(bootstrap_ates) == 0:
            print(f"警告：所有 {n_bootstrap} 次bootstrap都失败了，无法计算置信区间")
            return (np.nan, np.nan)
        elif len(bootstrap_ates) < n_bootstrap * 0.5:
            print(f"警告：只有 {len(bootstrap_ates)}/{n_bootstrap} 次bootstrap成功，置信区间可能不可靠")
        
        bootstrap_ates = np.array(bootstrap_ates)
        lower = np.percentile(bootstrap_ates, 100 * alpha / 2)
        upper = np.percentile(bootstrap_ates, 100 * (1 - alpha / 2))
        
        return (lower, upper)
    
    @abstractmethod
    def _estimate_ate(self, X, w, y):
        """内部ATE估计方法，供bootstrap使用"""
        pass
    
    def estimate_conditional_effects(self, X, w, y):
        """估计条件处理效应 (CATE)"""
        raise NotImplementedError("子类需要实现此方法以支持异质性处理效应")


class MLModelMixin:
    """机器学习模型混入类，提供模型选择和验证功能"""
    
    def __init__(self):
        self.supported_models = {
            'logistic': self._get_logistic_regression,
            'random_forest': self._get_random_forest,
            'gradient_boosting': self._get_gradient_boosting,
            'xgboost': self._get_xgboost,
            'neural_network': self._get_neural_network
        }
    
    def _get_logistic_regression(self, task='classification'):
        from sklearn.linear_model import LogisticRegression, LinearRegression
        if task == 'classification':
            return LogisticRegression(solver='liblinear', random_state=self.random_state)
        else:
            return LinearRegression()
    
    def _get_random_forest(self, task='classification'):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        if task == 'classification':
            return RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        else:
            return RandomForestRegressor(n_estimators=100, random_state=self.random_state)
    
    def _get_gradient_boosting(self, task='classification'):
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        if task == 'classification':
            return GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)
        else:
            return GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)
    
    def _get_xgboost(self, task='classification'):
        try:
            import xgboost as xgb
            if task == 'classification':
                return xgb.XGBClassifier(n_estimators=100, random_state=self.random_state)
            else:
                return xgb.XGBRegressor(n_estimators=100, random_state=self.random_state)
        except ImportError:
            print("XGBoost not available, falling back to GradientBoosting")
            return self._get_gradient_boosting(task)
    
    def _get_neural_network(self, task='classification'):
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        if task == 'classification':
            return MLPClassifier(hidden_layer_sizes=(50,), random_state=self.random_state, 
                               max_iter=100, early_stopping=True, validation_fraction=0.1)
        else:
            return MLPRegressor(hidden_layer_sizes=(50,), random_state=self.random_state, 
                              max_iter=100, early_stopping=True, validation_fraction=0.1)
    
    def get_model(self, model_name, task='classification'):
        """获取指定的机器学习模型"""
        if model_name not in self.supported_models:
            raise ValueError(f"不支持的模型: {model_name}. 支持的模型: {list(self.supported_models.keys())}")
        
        return self.supported_models[model_name](task)
    
    def select_best_model(self, models, X, y, cv=5, scoring=None):
        """通过交叉验证选择最佳模型"""
        best_score = -np.inf
        best_model = None
        
        for model_name, model in models.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
                mean_score = np.mean(scores)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = clone(model)
                    
                print(f"{model_name}: {mean_score:.4f} (+/- {np.std(scores) * 2:.4f})")
            except Exception as e:
                print(f"{model_name} failed: {e}")
                continue
        
        return best_model, best_score 