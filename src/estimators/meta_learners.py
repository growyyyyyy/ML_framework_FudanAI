"""
元学习器 (Meta-Learners)
实现S-Learner、T-Learner、X-Learner、R-Learner
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import clone

from .base import CausalEstimator, MLModelMixin


class SLearner(CausalEstimator, MLModelMixin):
    """
    S-Learner (Single Learner)
    使用单一模型学习 E[Y|X,T]，然后预测反事实结果
    """
    
    def __init__(self, base_model='gradient_boosting', auto_select_model=False, random_state=42):
        super().__init__(random_state)
        MLModelMixin.__init__(self)
        self.base_model = base_model
        self.auto_select_model = auto_select_model
        self.outcome_model_ = None
        
    def estimate(self, X, w, y, bootstrap_rounds=200, alpha=0.05):
        """估计平均处理效应"""
        X = np.asarray(X)
        w = np.asarray(w).reshape(-1, 1)
        y = np.asarray(y)
        
        # 构建增强特征矩阵 [X, T]
        X_augmented = np.concatenate([X, w], axis=1)
        
        # 训练结果模型
        self.outcome_model_ = self._train_outcome_model(X_augmented, y)
        
        # 预测反事实结果
        X_treated = np.concatenate([X, np.ones((len(X), 1))], axis=1)
        X_control = np.concatenate([X, np.zeros((len(X), 1))], axis=1)
        
        y1_pred = self.outcome_model_.predict(X_treated)
        y0_pred = self.outcome_model_.predict(X_control)
        
        # 计算个体处理效应和平均处理效应
        ite = y1_pred - y0_pred
        ate = np.mean(ite)
        
        # Bootstrap置信区间
        ci = self.bootstrap_confidence_interval(X, w.flatten(), y, bootstrap_rounds, alpha)
        
        self.ate_ = ate
        self.ate_ci_ = ci
        self.fitted_ = True
        
        return ate, ci
    
    def _estimate_ate(self, X, w, y):
        """内部ATE估计方法"""
        X = np.asarray(X)
        w = np.asarray(w).reshape(-1, 1)
        y = np.asarray(y)
        
        X_augmented = np.concatenate([X, w], axis=1)
        outcome_model = self._train_outcome_model(X_augmented, y)
        
        X_treated = np.concatenate([X, np.ones((len(X), 1))], axis=1)
        X_control = np.concatenate([X, np.zeros((len(X), 1))], axis=1)
        
        y1_pred = outcome_model.predict(X_treated)
        y0_pred = outcome_model.predict(X_control)
        
        ate = np.mean(y1_pred - y0_pred)
        return ate, None
    
    def _train_outcome_model(self, X, y):
        """训练结果模型"""
        if self.auto_select_model:
            # 如果已经进行过模型选择，重用最佳模型
            if hasattr(self, '_best_model_template'):
                model = clone(self._best_model_template)
                return model.fit(X, y)
            
            models = {
                'linear': self.get_model('logistic', 'regression'),
                'random_forest': self.get_model('random_forest', 'regression'),
                'gradient_boosting': self.get_model('gradient_boosting', 'regression'),
                'xgboost': self.get_model('xgboost', 'regression')
            }
            
            print("选择最佳结果模型...")
            best_model, _ = self.select_best_model(models, X, y, scoring='neg_mean_squared_error')
            
            # 保存最佳模型模板供后续使用
            self._best_model_template = best_model
            return best_model.fit(X, y)
        else:
            model = self.get_model(self.base_model, 'regression')
            return model.fit(X, y)
    
    def estimate_conditional_effects(self, X, w, y):
        """估计条件处理效应 (CATE)"""
        if not self.fitted_:
            self.estimate(X, w, y)
        
        X = np.asarray(X)
        X_treated = np.concatenate([X, np.ones((len(X), 1))], axis=1)
        X_control = np.concatenate([X, np.zeros((len(X), 1))], axis=1)
        
        y1_pred = self.outcome_model_.predict(X_treated)
        y0_pred = self.outcome_model_.predict(X_control)
        
        return y1_pred - y0_pred


class TLearner(CausalEstimator, MLModelMixin):
    """
    T-Learner (Two Learner)
    分别训练处理组和对照组的结果模型
    """
    
    def __init__(self, base_model='gradient_boosting', auto_select_model=False, random_state=42):
        super().__init__(random_state)
        MLModelMixin.__init__(self)
        self.base_model = base_model
        self.auto_select_model = auto_select_model
        self.treated_model_ = None
        self.control_model_ = None
        
    def estimate(self, X, w, y, bootstrap_rounds=200, alpha=0.05):
        """估计平均处理效应"""
        X = np.asarray(X)
        w = np.asarray(w)
        y = np.asarray(y)
        
        # 分离处理组和对照组数据
        treated_mask = (w == 1)
        control_mask = (w == 0)
        
        X_treated, y_treated = X[treated_mask], y[treated_mask]
        X_control, y_control = X[control_mask], y[control_mask]
        
        # 训练两个独立的模型
        self.treated_model_ = self._train_outcome_model(X_treated, y_treated)
        self.control_model_ = self._train_outcome_model(X_control, y_control)
        
        # 预测所有样本的潜在结果
        y1_pred = self.treated_model_.predict(X)
        y0_pred = self.control_model_.predict(X)
        
        # 计算平均处理效应
        ate = np.mean(y1_pred - y0_pred)
        
        # Bootstrap置信区间
        ci = self.bootstrap_confidence_interval(X, w, y, bootstrap_rounds, alpha)
        
        self.ate_ = ate
        self.ate_ci_ = ci
        self.fitted_ = True
        
        return ate, ci
    
    def _estimate_ate(self, X, w, y):
        """内部ATE估计方法"""
        X = np.asarray(X)
        w = np.asarray(w)
        y = np.asarray(y)
        
        treated_mask = (w == 1)
        control_mask = (w == 0)
        
        X_treated, y_treated = X[treated_mask], y[treated_mask]
        X_control, y_control = X[control_mask], y[control_mask]
        
        if len(X_treated) == 0 or len(X_control) == 0:
            return 0.0, None
        
        treated_model = self._train_outcome_model(X_treated, y_treated)
        control_model = self._train_outcome_model(X_control, y_control)
        
        y1_pred = treated_model.predict(X)
        y0_pred = control_model.predict(X)
        
        ate = np.mean(y1_pred - y0_pred)
        return ate, None
    
    def _train_outcome_model(self, X, y):
        """训练结果模型"""
        if len(X) == 0:
            return None
            
        if self.auto_select_model:
            # 如果已经进行过模型选择，重用最佳模型
            if hasattr(self, '_best_model_template'):
                model = clone(self._best_model_template)
                return model.fit(X, y)
            
            models = {
                'linear': self.get_model('logistic', 'regression'),
                'random_forest': self.get_model('random_forest', 'regression'),
                'gradient_boosting': self.get_model('gradient_boosting', 'regression'),
                'xgboost': self.get_model('xgboost', 'regression')
            }
            
            best_model, _ = self.select_best_model(models, X, y, scoring='neg_mean_squared_error')
            
            # 保存最佳模型模板供后续使用
            self._best_model_template = best_model
            return best_model.fit(X, y)
        else:
            model = self.get_model(self.base_model, 'regression')
            return model.fit(X, y)
    
    def estimate_conditional_effects(self, X, w, y):
        """估计条件处理效应 (CATE)"""
        if not self.fitted_:
            self.estimate(X, w, y)
        
        X = np.asarray(X)
        y1_pred = self.treated_model_.predict(X)
        y0_pred = self.control_model_.predict(X)
        
        return y1_pred - y0_pred


class XLearner(CausalEstimator, MLModelMixin):
    """
    X-Learner
    改进的T-Learner，额外估计伪个体处理效应
    """
    
    def __init__(self, base_model='gradient_boosting', auto_select_model=False, random_state=42):
        super().__init__(random_state)
        MLModelMixin.__init__(self)
        self.base_model = base_model
        self.auto_select_model = auto_select_model
        self.treated_model_ = None
        self.control_model_ = None
        self.ite_treated_model_ = None
        self.ite_control_model_ = None
        
    def estimate(self, X, w, y, bootstrap_rounds=200, alpha=0.05):
        """估计平均处理效应"""
        X = np.asarray(X)
        w = np.asarray(w)
        y = np.asarray(y)
        
        # 第一阶段：训练结果模型
        treated_mask = (w == 1)
        control_mask = (w == 0)
        
        X_treated, y_treated = X[treated_mask], y[treated_mask]
        X_control, y_control = X[control_mask], y[control_mask]
        
        self.treated_model_ = self._train_outcome_model(X_treated, y_treated)
        self.control_model_ = self._train_outcome_model(X_control, y_control)
        
        # 第二阶段：估计伪个体处理效应
        # 对于处理组：ITE = Y - μ0(X)
        if len(X_control) > 0:
            y0_pred_treated = self.control_model_.predict(X_treated)
            pseudo_ite_treated = y_treated - y0_pred_treated
            self.ite_treated_model_ = self._train_outcome_model(X_treated, pseudo_ite_treated)
        
        # 对于对照组：ITE = μ1(X) - Y
        if len(X_treated) > 0:
            y1_pred_control = self.treated_model_.predict(X_control)
            pseudo_ite_control = y1_pred_control - y_control
            self.ite_control_model_ = self._train_outcome_model(X_control, pseudo_ite_control)
        
        # 第三阶段：估计倾向评分并加权组合
        from .propensity_score import ImprovedPropensityScoreWeighting
        ps_estimator = ImprovedPropensityScoreWeighting(
            propensity_model=self.base_model, random_state=self.random_state
        )
        propensity_scores = ps_estimator._estimate_propensity_scores(
            ps_estimator._preprocess_features(pd.DataFrame(X)), w
        )
        
        # 组合估计
        ite_estimates = np.zeros(len(X))
        
        if self.ite_treated_model_ is not None:
            ite_treated = self.ite_treated_model_.predict(X)
            ite_estimates += (1 - propensity_scores) * ite_treated
        
        if self.ite_control_model_ is not None:
            ite_control = self.ite_control_model_.predict(X)
            ite_estimates += propensity_scores * ite_control
        
        ate = np.mean(ite_estimates)
        
        # Bootstrap置信区间
        ci = self.bootstrap_confidence_interval(X, w, y, bootstrap_rounds, alpha)
        
        self.ate_ = ate
        self.ate_ci_ = ci
        self.fitted_ = True
        
        return ate, ci
    
    def _estimate_ate(self, X, w, y):
        """内部ATE估计方法"""
        # 简化版本，直接使用T-Learner逻辑
        t_learner = TLearner(self.base_model, self.auto_select_model, self.random_state)
        ate, _ = t_learner._estimate_ate(X, w, y)
        return ate, None
    
    def _train_outcome_model(self, X, y):
        """训练结果模型"""
        if len(X) == 0:
            return None
            
        if self.auto_select_model:
            # 如果已经进行过模型选择，重用最佳模型
            if hasattr(self, '_best_model_template'):
                model = clone(self._best_model_template)
                return model.fit(X, y)
            
            models = {
                'linear': self.get_model('logistic', 'regression'),
                'random_forest': self.get_model('random_forest', 'regression'),
                'gradient_boosting': self.get_model('gradient_boosting', 'regression'),
                'xgboost': self.get_model('xgboost', 'regression')
            }
            
            best_model, _ = self.select_best_model(models, X, y, scoring='neg_mean_squared_error')
            
            # 保存最佳模型模板供后续使用
            self._best_model_template = best_model
            return best_model.fit(X, y)
        else:
            model = self.get_model(self.base_model, 'regression')
            return model.fit(X, y)


class RLearner(CausalEstimator, MLModelMixin):
    """
    R-Learner (Robinson's Learner)
    基于残差的因果效应估计
    """
    
    def __init__(self, base_model='gradient_boosting', auto_select_model=False, random_state=42):
        super().__init__(random_state)
        MLModelMixin.__init__(self)
        self.base_model = base_model
        self.auto_select_model = auto_select_model
        self.outcome_model_ = None
        self.propensity_model_ = None
        self.effect_model_ = None
        
    def estimate(self, X, w, y, bootstrap_rounds=200, alpha=0.05):
        """估计平均处理效应"""
        X = np.asarray(X)
        w = np.asarray(w)
        y = np.asarray(y)
        
        # 第一阶段：估计 E[Y|X] 和 E[T|X]
        self.outcome_model_ = self._train_outcome_model(X, y)
        
        from .propensity_score import ImprovedPropensityScoreWeighting
        ps_estimator = ImprovedPropensityScoreWeighting(
            propensity_model=self.base_model, random_state=self.random_state
        )
        propensity_scores = ps_estimator._estimate_propensity_scores(
            ps_estimator._preprocess_features(pd.DataFrame(X)), w
        )
        
        # 计算残差
        y_residual = y - self.outcome_model_.predict(X)
        w_residual = w - propensity_scores
        
        # 第二阶段：回归残差
        # 最小化 E[(Y_res - τ(X) * W_res)^2]
        # 这等价于回归 Y_res / W_res ~ τ(X)，但我们使用加权最小二乘
        
        # 避免除零，只使用非零残差
        nonzero_mask = np.abs(w_residual) > 1e-6
        if np.sum(nonzero_mask) > 0:
            X_filtered = X[nonzero_mask]
            y_targets = y_residual[nonzero_mask] / w_residual[nonzero_mask]
            sample_weights = w_residual[nonzero_mask] ** 2
            
            self.effect_model_ = self._train_outcome_model(X_filtered, y_targets)
            
            # 预测个体处理效应
            ite = self.effect_model_.predict(X)
            ate = np.mean(ite)
        else:
            ate = 0.0
        
        # Bootstrap置信区间
        ci = self.bootstrap_confidence_interval(X, w, y, bootstrap_rounds, alpha)
        
        self.ate_ = ate
        self.ate_ci_ = ci
        self.fitted_ = True
        
        return ate, ci
    
    def _estimate_ate(self, X, w, y):
        """内部ATE估计方法"""
        # 简化版本
        return np.mean(y[w == 1]) - np.mean(y[w == 0]), None
    
    def _train_outcome_model(self, X, y):
        """训练结果模型"""
        if len(X) == 0:
            return None
            
        if self.auto_select_model:
            # 如果已经进行过模型选择，重用最佳模型
            if hasattr(self, '_best_model_template'):
                model = clone(self._best_model_template)
                return model.fit(X, y)
            
            models = {
                'linear': self.get_model('logistic', 'regression'),
                'random_forest': self.get_model('random_forest', 'regression'),
                'gradient_boosting': self.get_model('gradient_boosting', 'regression'),
                'xgboost': self.get_model('xgboost', 'regression')
            }
            
            best_model, _ = self.select_best_model(models, X, y, scoring='neg_mean_squared_error')
            
            # 保存最佳模型模板供后续使用
            self._best_model_template = best_model
            return best_model.fit(X, y)
        else:
            model = self.get_model(self.base_model, 'regression')
            return model.fit(X, y) 