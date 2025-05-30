"""
因果森林 (Causal Forest)
简化版实现，基于随机森林的异质性处理效应估计
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from .base import CausalEstimator, MLModelMixin


class CausalForestEstimator(CausalEstimator, MLModelMixin):
    """
    因果森林估计器
    
    简化版实现，基于修改的随机森林算法
    能够估计异质性处理效应 (Heterogeneous Treatment Effects)
    """
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=10,
                 min_samples_leaf=5, honesty_fraction=0.5, random_state=42):
        super().__init__(random_state)
        MLModelMixin.__init__(self)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.honesty_fraction = honesty_fraction
        self.trees_ = []
        self.feature_importances_ = None
        
    def estimate(self, X, w, y, bootstrap_rounds=200, alpha=0.05):
        """估计平均处理效应"""
        X = np.asarray(X)
        w = np.asarray(w)
        y = np.asarray(y)
        
        # 训练因果森林
        self._fit_causal_forest(X, w, y)
        
        # 预测个体处理效应
        individual_effects = self.predict_treatment_effects(X)
        
        # 计算平均处理效应
        ate = np.mean(individual_effects)
        
        # Bootstrap置信区间
        ci = self.bootstrap_confidence_interval(X, w, y, bootstrap_rounds, alpha)
        
        self.ate_ = ate
        self.ate_ci_ = ci
        self.fitted_ = True
        
        return ate, ci
    
    def _estimate_ate(self, X, w, y):
        """内部ATE估计方法"""
        # 简单版本：训练单棵因果树
        causal_tree = self._fit_single_causal_tree(X, w, y)
        individual_effects = causal_tree.predict_treatment_effects(X)
        ate = np.mean(individual_effects)
        return ate, None
    
    def _fit_causal_forest(self, X, w, y):
        """训练因果森林"""
        np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        self.trees_ = []
        
        for i in range(self.n_estimators):
            # Bootstrap抽样
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            w_boot = w[indices]
            y_boot = y[indices]
            
            # 特征抽样
            feature_indices = np.random.choice(
                n_features, 
                max(1, int(np.sqrt(n_features))), 
                replace=False
            )
            X_boot_selected = X_boot[:, feature_indices]
            
            # 训练单棵因果树
            tree = CausalTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                honesty_fraction=self.honesty_fraction,
                feature_indices=feature_indices,
                random_state=self.random_state + i
            )
            
            tree.fit(X_boot_selected, w_boot, y_boot)
            self.trees_.append(tree)
        
        # 计算特征重要性（简化版）
        self.feature_importances_ = np.zeros(n_features)
        for tree in self.trees_:
            if hasattr(tree, 'feature_importances_'):
                for idx, importance in zip(tree.feature_indices, tree.feature_importances_):
                    self.feature_importances_[idx] += importance
        
        self.feature_importances_ /= self.n_estimators
    
    def _fit_single_causal_tree(self, X, w, y):
        """训练单棵因果树"""
        tree = CausalTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            honesty_fraction=self.honesty_fraction,
            random_state=self.random_state
        )
        tree.fit(X, w, y)
        return tree
    
    def predict_treatment_effects(self, X):
        """预测个体处理效应"""
        if not self.fitted_:
            raise ValueError("模型未拟合，请先调用estimate()方法")
        
        X = np.asarray(X)
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, len(self.trees_)))
        
        for i, tree in enumerate(self.trees_):
            X_selected = X[:, tree.feature_indices]
            predictions[:, i] = tree.predict_treatment_effects(X_selected)
        
        # 返回所有树的平均预测
        return np.mean(predictions, axis=1)
    
    def estimate_conditional_effects(self, X, w, y):
        """估计条件处理效应 (CATE)"""
        if not self.fitted_:
            self.estimate(X, w, y)
        
        return self.predict_treatment_effects(X)
    
    def variable_importance(self):
        """返回变量重要性"""
        if not self.fitted_:
            raise ValueError("模型未拟合，请先调用estimate()方法")
        
        return self.feature_importances_


class CausalTree:
    """
    单棵因果树
    
    基于诚实估计(Honest Estimation)的因果树
    """
    
    def __init__(self, max_depth=None, min_samples_split=10, min_samples_leaf=5,
                 honesty_fraction=0.5, feature_indices=None, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.honesty_fraction = honesty_fraction
        self.feature_indices = feature_indices
        self.random_state = random_state
        self.tree_ = None
        self.leaf_treatment_effects_ = {}
        
    def fit(self, X, w, y):
        """训练因果树"""
        np.random.seed(self.random_state)
        
        # 诚实分割：一部分数据用于分割，一部分用于估计叶节点效应
        n_samples = len(X)
        split_size = int(n_samples * (1 - self.honesty_fraction))
        
        indices = np.random.permutation(n_samples)
        split_indices = indices[:split_size]
        estimate_indices = indices[split_size:]
        
        # 分割阶段：使用分割数据构建树结构
        X_split, w_split, y_split = X[split_indices], w[split_indices], y[split_indices]
        
        # 简化：使用标准决策树进行分割，基于处理效应的方差
        # 这里我们创建增强特征来帮助分割
        interaction_outcomes = y_split * w_split  # 简单的交互项
        
        self.tree_ = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        
        self.tree_.fit(X_split, interaction_outcomes)
        
        # 估计阶段：使用估计数据计算叶节点的处理效应
        if len(estimate_indices) > 0:
            X_estimate, w_estimate, y_estimate = X[estimate_indices], w[estimate_indices], y[estimate_indices]
            self._estimate_leaf_effects(X_estimate, w_estimate, y_estimate)
        else:
            # 如果没有估计数据，使用分割数据
            self._estimate_leaf_effects(X_split, w_split, y_split)
    
    def _estimate_leaf_effects(self, X, w, y):
        """估计每个叶节点的处理效应"""
        # 获取叶节点分配
        leaf_indices = self.tree_.apply(X)
        unique_leaves = np.unique(leaf_indices)
        
        self.leaf_treatment_effects_ = {}
        
        for leaf_id in unique_leaves:
            leaf_mask = (leaf_indices == leaf_id)
            
            if np.sum(leaf_mask) == 0:
                continue
            
            y_leaf = y[leaf_mask]
            w_leaf = w[leaf_mask]
            
            # 计算叶节点内的处理效应
            treated_mask = (w_leaf == 1)
            control_mask = (w_leaf == 0)
            
            if np.sum(treated_mask) > 0 and np.sum(control_mask) > 0:
                # 有处理组和对照组数据
                ate_leaf = np.mean(y_leaf[treated_mask]) - np.mean(y_leaf[control_mask])
            elif np.sum(treated_mask) > 0:
                # 只有处理组数据，使用全局对照均值
                global_control_mean = np.mean(y[w == 0]) if np.sum(w == 0) > 0 else 0
                ate_leaf = np.mean(y_leaf[treated_mask]) - global_control_mean
            elif np.sum(control_mask) > 0:
                # 只有对照组数据，使用全局处理均值
                global_treated_mean = np.mean(y[w == 1]) if np.sum(w == 1) > 0 else 0
                ate_leaf = global_treated_mean - np.mean(y_leaf[control_mask])
            else:
                # 异常情况
                ate_leaf = 0.0
            
            self.leaf_treatment_effects_[leaf_id] = ate_leaf
    
    def predict_treatment_effects(self, X):
        """预测处理效应"""
        if self.tree_ is None:
            raise ValueError("树未训练，请先调用fit()方法")
        
        leaf_indices = self.tree_.apply(X)
        predictions = np.zeros(len(X))
        
        for i, leaf_id in enumerate(leaf_indices):
            if leaf_id in self.leaf_treatment_effects_:
                predictions[i] = self.leaf_treatment_effects_[leaf_id]
            else:
                # 如果叶节点没有效应估计，使用默认值0
                predictions[i] = 0.0
        
        return predictions 