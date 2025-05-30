"""
双重机器学习 (Double Machine Learning)
实现DML的无偏因果效应估计
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.base import clone

from .base import CausalEstimator, MLModelMixin


class DoubleMLEstimator(CausalEstimator, MLModelMixin):
    """
    双重机器学习估计器
    
    同时建模处理变量和结果变量，使用交叉拟合来减少正则化偏差
    基于 Chernozhukov et al. (2018) 的方法
    """
    
    def __init__(self, outcome_model='gradient_boosting', propensity_model='gradient_boosting',
                 n_folds=5, auto_select_model=False, random_state=42):
        super().__init__(random_state)
        MLModelMixin.__init__(self)
        self.outcome_model = outcome_model
        self.propensity_model = propensity_model
        self.n_folds = n_folds
        self.auto_select_model = auto_select_model
        self.outcome_models_ = []
        self.propensity_models_ = []
        self.theta_scores_ = []
        
    def estimate(self, X, w, y, bootstrap_rounds=200, alpha=0.05):
        """估计平均处理效应"""
        X = np.asarray(X)
        w = np.asarray(w)
        y = np.asarray(y)
        
        # 交叉拟合
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # 存储每折的预测残差
        y_residuals = np.zeros_like(y)
        w_residuals = np.zeros_like(w, dtype=float)
        
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            w_train, w_test = w[train_idx], w[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 训练结果模型 E[Y|X]
            outcome_model = self._get_outcome_model()
            outcome_model.fit(X_train, y_train)
            self.outcome_models_.append(outcome_model)
            
            # 训练倾向评分模型 E[T|X]
            propensity_model = self._get_propensity_model()
            propensity_model.fit(X_train, w_train)
            self.propensity_models_.append(propensity_model)
            
            # 预测测试集
            y_pred = outcome_model.predict(X_test)
            
            if hasattr(propensity_model, 'predict_proba'):
                w_pred = propensity_model.predict_proba(X_test)[:, 1]
            else:
                w_pred = propensity_model.predict(X_test)
            
            # 计算残差
            y_residuals[test_idx] = y_test - y_pred
            w_residuals[test_idx] = w_test - w_pred
        
        # DML的矩方程估计
        # θ = E[ψ(W,Y,X;θ)] = 0
        # 其中 ψ(W,Y,X;θ) = (Y - l(X)) - θ(W - m(X))
        # 解得 θ = E[(Y - l(X))(W - m(X))] / E[(W - m(X))^2]
        
        numerator = np.mean(y_residuals * w_residuals)
        denominator = np.mean(w_residuals ** 2)
        
        if abs(denominator) > 1e-8:
            ate = numerator / denominator
        else:
            # 如果分母接近0，退回到简单差分
            ate = np.mean(y[w == 1]) - np.mean(y[w == 0])
        
        # 计算渐近方差和置信区间
        # 根据DML理论，使用影响函数计算标准误
        influence_function = self._compute_influence_function(y_residuals, w_residuals, ate)
        variance = np.var(influence_function) / len(influence_function)
        std_error = np.sqrt(variance)
        
        # 正态分布置信区间
        from scipy.stats import norm
        critical_value = norm.ppf(1 - alpha / 2)
        ci = (ate - critical_value * std_error, ate + critical_value * std_error)
        
        self.ate_ = ate
        self.ate_ci_ = ci
        self.fitted_ = True
        
        return ate, ci
    
    def _estimate_ate(self, X, w, y):
        """内部ATE估计方法"""
        # 简化版本，不使用交叉拟合
        outcome_model = self._get_outcome_model()
        propensity_model = self._get_propensity_model()
        
        outcome_model.fit(X, y)
        propensity_model.fit(X, w)
        
        y_pred = outcome_model.predict(X)
        if hasattr(propensity_model, 'predict_proba'):
            w_pred = propensity_model.predict_proba(X)[:, 1]
        else:
            w_pred = propensity_model.predict(X)
        
        y_residuals = y - y_pred
        w_residuals = w - w_pred
        
        numerator = np.mean(y_residuals * w_residuals)
        denominator = np.mean(w_residuals ** 2)
        
        if abs(denominator) > 1e-8:
            ate = numerator / denominator
        else:
            ate = np.mean(y[w == 1]) - np.mean(y[w == 0])
        
        return ate, None
    
    def _get_outcome_model(self):
        """获取结果模型"""
        if self.auto_select_model:
            models = {
                'linear': self.get_model('logistic', 'regression'),
                'random_forest': self.get_model('random_forest', 'regression'),
                'gradient_boosting': self.get_model('gradient_boosting', 'regression'),
                'xgboost': self.get_model('xgboost', 'regression')
            }
            # 这里简化，实际应该在每折中选择
            best_model, _ = self.select_best_model(models, None, None)
            return clone(best_model)
        else:
            return self.get_model(self.outcome_model, 'regression')
    
    def _get_propensity_model(self):
        """获取倾向评分模型"""
        if self.auto_select_model:
            models = {
                'logistic': self.get_model('logistic', 'classification'),
                'random_forest': self.get_model('random_forest', 'classification'),
                'gradient_boosting': self.get_model('gradient_boosting', 'classification'),
                'xgboost': self.get_model('xgboost', 'classification')
            }
            # 这里简化，实际应该在每折中选择
            best_model, _ = self.select_best_model(models, None, None)
            return clone(best_model)
        else:
            return self.get_model(self.propensity_model, 'classification')
    
    def _compute_influence_function(self, y_residuals, w_residuals, theta):
        """计算影响函数用于方差估计"""
        # 影响函数：IF(θ) = (Y_res - θ * W_res) * W_res / E[W_res^2]
        denominator = np.mean(w_residuals ** 2)
        if abs(denominator) > 1e-8:
            influence_scores = (y_residuals - theta * w_residuals) * w_residuals / denominator
        else:
            influence_scores = np.zeros_like(y_residuals)
        
        return influence_scores
    
    def estimate_conditional_effects(self, X, w, y):
        """
        估计条件处理效应 (CATE)
        注意：标准DML主要估计ATE，CATE需要额外的建模
        """
        if not self.fitted_:
            self.estimate(X, w, y)
        
        # 简化实现：假设处理效应是常数
        cate = np.full(len(X), self.ate_)
        return cate
    
    def get_residuals(self, X, w, y):
        """获取最后一次估计的残差（用于诊断）"""
        if not self.fitted_:
            raise ValueError("模型未拟合，请先调用estimate()方法")
        
        # 重新计算残差（简化版本）
        outcome_model = self.outcome_models_[-1] if self.outcome_models_ else self._get_outcome_model()
        propensity_model = self.propensity_models_[-1] if self.propensity_models_ else self._get_propensity_model()
        
        if not hasattr(outcome_model, 'predict'):
            outcome_model.fit(X, y)
            propensity_model.fit(X, w)
        
        y_pred = outcome_model.predict(X)
        if hasattr(propensity_model, 'predict_proba'):
            w_pred = propensity_model.predict_proba(X)[:, 1]
        else:
            w_pred = propensity_model.predict(X)
        
        return {
            'y_residuals': y - y_pred,
            'w_residuals': w - w_pred,
            'y_predictions': y_pred,
            'w_predictions': w_pred
        } 