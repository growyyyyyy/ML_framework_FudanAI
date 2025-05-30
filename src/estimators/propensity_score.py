"""
倾向评分估计器
支持多种机器学习算法和模型选择
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from .base import CausalEstimator, MLModelMixin


class PropensityScoreWeighting(CausalEstimator):
    """原始的倾向评分加权估计器（向后兼容）"""
    
    def estimate(self, X, w, y, bootstrap_rounds=200, alpha=0.05):
        # 保持原有接口不变
        ps = LogisticRegression(solver='liblinear').fit(X, w).predict_proba(X)[:,1]
        wt_t = w / ps
        wt_c = (1-w) / (1-ps)
        ate = np.average(y[w==1], weights=wt_t[w==1]) - np.average(y[w==0], weights=wt_c[w==0])
        
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
    
    def _estimate_ate(self, X, w, y):
        ps = LogisticRegression(solver='liblinear').fit(X, w).predict_proba(X)[:,1]
        wt_t = w / ps
        wt_c = (1-w) / (1-ps)
        ate = np.average(y[w==1], weights=wt_t[w==1]) - np.average(y[w==0], weights=wt_c[w==0])
        return ate, None


class ImprovedPropensityScoreWeighting(CausalEstimator, MLModelMixin):
    """改进的倾向评分加权估计器，支持多种ML算法"""
    
    def __init__(self, propensity_model='logistic', auto_select_model=False, 
                 trim_threshold=0.05, normalize_weights=True, random_state=42):
        super().__init__(random_state)
        MLModelMixin.__init__(self)
        self.propensity_model = propensity_model
        self.auto_select_model = auto_select_model
        self.trim_threshold = trim_threshold
        self.normalize_weights = normalize_weights
        self.propensity_estimator_ = None
        self.propensity_scores_ = None
        
    def estimate(self, X, w, y, bootstrap_rounds=200, alpha=0.05):
        """估计平均处理效应"""
        # 预处理数据
        X = self._preprocess_features(X)
        w = np.asarray(w)
        y = np.asarray(y)
        
        # 估计倾向评分
        self.propensity_scores_ = self._estimate_propensity_scores(X, w)
        
        # 计算权重
        weights = self._compute_weights(w, self.propensity_scores_)
        
        # 估计ATE
        ate = self._compute_weighted_ate(w, y, weights)
        
        # Bootstrap置信区间
        ci = self.bootstrap_confidence_interval(X, w, y, bootstrap_rounds, alpha)
        
        self.ate_ = ate
        self.ate_ci_ = ci
        self.fitted_ = True
        
        return ate, ci
    
    def _estimate_ate(self, X, w, y):
        """内部ATE估计方法"""
        X = self._preprocess_features(X)
        
        # 如果已经有训练好的模型，重用它而不是重新选择
        if hasattr(self, 'propensity_estimator_') and self.propensity_estimator_ is not None:
            # 重用已选择的最佳模型，但用新数据重新训练
            estimator = clone(self.propensity_estimator_)
            estimator.fit(X, w)
            
            if hasattr(estimator, 'predict_proba'):
                propensity_scores = estimator.predict_proba(X)[:, 1]
            else:
                propensity_scores = estimator.decision_function(X)
                propensity_scores = 1 / (1 + np.exp(-propensity_scores))
        else:
            # 如果没有预训练的模型，进行完整的估计过程
            propensity_scores = self._estimate_propensity_scores(X, w)
        
        weights = self._compute_weights(w, propensity_scores)
        ate = self._compute_weighted_ate(w, y, weights)
        return ate, None
    
    def _preprocess_features(self, X):
        """特征预处理"""
        if hasattr(X, 'values'):
            X = X.values
        
        # 标准化特征
        if not hasattr(self, 'scaler_'):
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = self.scaler_.transform(X)
        
        return X_scaled
    
    def _estimate_propensity_scores(self, X, w):
        """估计倾向评分"""
        if self.auto_select_model:
            # 自动选择最佳模型
            models = {
                'logistic': self.get_model('logistic', 'classification'),
                'random_forest': self.get_model('random_forest', 'classification'),
                'gradient_boosting': self.get_model('gradient_boosting', 'classification'),
                'xgboost': self.get_model('xgboost', 'classification')
            }
            
            print("选择最佳倾向评分模型...")
            best_model, best_score = self.select_best_model(models, X, w, scoring='roc_auc')
            self.propensity_estimator_ = best_model
        else:
            # 使用指定模型
            self.propensity_estimator_ = self.get_model(self.propensity_model, 'classification')
        
        # 拟合模型
        self.propensity_estimator_.fit(X, w)
        
        # 预测倾向评分
        if hasattr(self.propensity_estimator_, 'predict_proba'):
            propensity_scores = self.propensity_estimator_.predict_proba(X)[:, 1]
        else:
            propensity_scores = self.propensity_estimator_.decision_function(X)
            # 转换为概率
            propensity_scores = 1 / (1 + np.exp(-propensity_scores))
        
        return propensity_scores
    
    def _compute_weights(self, w, propensity_scores):
        """计算倾向评分权重"""
        # 截断极端倾向评分
        if self.trim_threshold > 0:
            propensity_scores = np.clip(propensity_scores, 
                                      self.trim_threshold, 
                                      1 - self.trim_threshold)
        
        # 计算IPW权重
        weights = np.zeros_like(w, dtype=float)
        treated_mask = (w == 1)
        control_mask = (w == 0)
        
        weights[treated_mask] = 1 / propensity_scores[treated_mask]
        weights[control_mask] = 1 / (1 - propensity_scores[control_mask])
        
        # 权重归一化
        if self.normalize_weights:
            weights[treated_mask] /= np.sum(weights[treated_mask])
            weights[control_mask] /= np.sum(weights[control_mask])
            weights[treated_mask] *= np.sum(treated_mask)
            weights[control_mask] *= np.sum(control_mask)
        
        return weights
    
    def _compute_weighted_ate(self, w, y, weights):
        """计算加权平均处理效应"""
        treated_mask = (w == 1)
        control_mask = (w == 0)
        
        weighted_treated_outcome = np.average(y[treated_mask], weights=weights[treated_mask])
        weighted_control_outcome = np.average(y[control_mask], weights=weights[control_mask])
        
        ate = weighted_treated_outcome - weighted_control_outcome
        return ate
    
    def diagnostics(self, X, w):
        """倾向评分诊断"""
        if not self.fitted_:
            raise ValueError("模型未拟合，请先调用estimate()方法")
        
        results = {}
        
        # 倾向评分分布
        treated_scores = self.propensity_scores_[w == 1]
        control_scores = self.propensity_scores_[w == 0]
        
        results['propensity_score_summary'] = {
            'treated': {
                'mean': np.mean(treated_scores),
                'std': np.std(treated_scores),
                'min': np.min(treated_scores),
                'max': np.max(treated_scores)
            },
            'control': {
                'mean': np.mean(control_scores),
                'std': np.std(control_scores),
                'min': np.min(control_scores),
                'max': np.max(control_scores)
            }
        }
        
        # 重叠度检查
        overlap_min = max(np.min(treated_scores), np.min(control_scores))
        overlap_max = min(np.max(treated_scores), np.max(control_scores))
        results['overlap'] = {
            'range': (overlap_min, overlap_max),
            'proportion_in_overlap': np.mean(
                (self.propensity_scores_ >= overlap_min) & 
                (self.propensity_scores_ <= overlap_max)
            )
        }
        
        # 极端权重检查
        weights = self._compute_weights(w, self.propensity_scores_)
        results['weight_summary'] = {
            'mean': np.mean(weights),
            'std': np.std(weights),
            'max': np.max(weights),
            'proportion_large_weights': np.mean(weights > 10)  # 权重大于10的比例
        }
        
        return results 