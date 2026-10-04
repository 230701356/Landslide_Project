import numpy as np
import pandas as pd

class StandardScalerNumPy:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.array(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        X = np.array(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class LogisticRegressionNumPy:
    def __init__(self, lr=0.05, max_iter=800, random_state=42):
        self.lr = lr
        self.max_iter = max_iter
        self.random_state = random_state
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.max_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = 1.0 / (1.0 + np.exp(-np.clip(linear_model, -25, 25)))

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        prob = 1.0 / (1.0 + np.exp(-np.clip(linear_model, -25, 25)))
        return np.column_stack((1 - prob, prob))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    @property
    def feature_importances_(self):
        abs_w = np.abs(self.weights)
        total = np.sum(abs_w)
        return abs_w / total if total > 0 else abs_w


class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTreeNumPy:
    def __init__(self, max_depth=10, min_samples_split=4, max_features=None, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self.feature_importances_ = None

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_features = X.shape[1]
        self.feature_importances_ = np.zeros(n_features)
        self.root = self._build_tree(X, y, depth=0)
        total_imp = np.sum(self.feature_importances_)
        if total_imp > 0:
            self.feature_importances_ /= total_imp
        return self

    def _gini(self, y):
        if len(y) == 0:
            return 0
        p1 = np.mean(y)
        return 1.0 - (p1**2 + (1 - p1)**2)

    def _best_split(self, X, y):
        best_gain = -1.0
        split_idx, split_thresh = None, None
        n_samples, n_features = X.shape
        parent_gini = self._gini(y)

        features = np.arange(n_features)
        if self.max_features is not None and self.max_features < n_features:
            features = np.random.choice(n_features, self.max_features, replace=False)

        for feat in features:
            X_column = X[:, feat]
            thresholds = np.unique(X_column)
            if len(thresholds) > 20:
                thresholds = np.percentile(thresholds, np.linspace(5, 95, 15))

            for thresh in thresholds:
                left_idxs = X_column <= thresh
                right_idxs = ~left_idxs

                if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
                    continue

                n_l, n_r = np.sum(left_idxs), np.sum(right_idxs)
                gini_l = self._gini(y[left_idxs])
                gini_r = self._gini(y[right_idxs])
                weighted_gini = (n_l / n_samples) * gini_l + (n_r / n_samples) * gini_r

                gain = parent_gini - weighted_gini
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat
                    split_thresh = thresh

        if best_gain > 0 and split_idx is not None:
            self.feature_importances_[split_idx] += best_gain * n_samples

        return split_idx, split_thresh

    def _build_tree(self, X, y, depth=0):
        n_samples = X.shape[0]
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_val = np.mean(y) if len(y) > 0 else 0.0
            return TreeNode(value=leaf_val)

        feat_idx, thresh = self._best_split(X, y)
        if feat_idx is None:
            return TreeNode(value=np.mean(y))

        left_idxs = X[:, feat_idx] <= thresh
        right_idxs = ~left_idxs

        left_child = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_child = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)
        return TreeNode(feature=feat_idx, threshold=thresh, left=left_child, right=right_child)

    def _traverse_tree(self, x, node):
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict_proba(self, X):
        probs = np.array([self._traverse_tree(x, self.root) for x in X])
        probs = np.clip(probs, 0.0, 1.0)
        return np.column_stack((1 - probs, probs))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class RandomForestNumPy:
    def __init__(self, n_trees=25, max_depth=8, min_samples_split=4, random_state=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        max_features = int(np.sqrt(n_features)) + 1
        self.trees = []
        self.feature_importances_ = np.zeros(n_features)

        for i in range(self.n_trees):
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            tree = DecisionTreeNumPy(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features,
                random_state=self.random_state + i
            )
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)
            self.feature_importances_ += tree.feature_importances_

        total_imp = np.sum(self.feature_importances_)
        if total_imp > 0:
            self.feature_importances_ /= total_imp
        return self

    def predict_proba(self, X):
        tree_probs = np.array([tree.predict_proba(X)[:, 1] for tree in self.trees])
        avg_prob = np.mean(tree_probs, axis=0)
        return np.column_stack((1 - avg_prob, avg_prob))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class XGBoostNumPy:
    def __init__(self, n_estimators=30, learning_rate=0.1, max_depth=5, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.base_pred = 0.0
        self.feature_importances_ = None

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        p0 = np.mean(y)
        self.base_pred = np.log(p0 / (1 - p0 + 1e-7))
        F = np.full(n_samples, self.base_pred)

        self.trees = []
        self.feature_importances_ = np.zeros(n_features)

        for i in range(self.n_estimators):
            p = 1.0 / (1.0 + np.exp(-F))
            residuals = y - p

            tree = DecisionTreeNumPy(
                max_depth=self.max_depth,
                min_samples_split=4,
                random_state=self.random_state + i
            )
            tree.fit(X, residuals)
            tree_pred = tree.predict_proba(X)[:, 1] - 0.5
            F += self.learning_rate * tree_pred
            self.trees.append(tree)
            self.feature_importances_ += tree.feature_importances_

        total_imp = np.sum(self.feature_importances_)
        if total_imp > 0:
            self.feature_importances_ /= total_imp
        return self

    def predict_proba(self, X):
        F = np.full(X.shape[0], self.base_pred)
        for tree in self.trees:
            tree_pred = tree.predict_proba(X)[:, 1] - 0.5
            F += self.learning_rate * tree_pred
        prob = 1.0 / (1.0 + np.exp(-np.clip(F, -25, 25)))
        return np.column_stack((1 - prob, prob))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def calculate_metrics(y_true, y_pred, y_prob):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    # ROC AUC & Curve points
    thresholds = np.linspace(0.0, 1.0, 50)
    tprs = []
    fprs = []
    
    pos_count = np.sum(y_true == 1)
    neg_count = np.sum(y_true == 0)

    for th in thresholds:
        th_pred = (y_prob >= th).astype(int)
        th_tp = np.sum((y_true == 1) & (th_pred == 1))
        th_fp = np.sum((y_true == 0) & (th_pred == 1))
        
        tpr = th_tp / pos_count if pos_count > 0 else 0
        fpr = th_fp / neg_count if neg_count > 0 else 0
        tprs.append(tpr)
        fprs.append(fpr)

    # Sort by FPR for trapezoidal AUC integral
    sorted_pairs = sorted(zip(fprs, tprs))
    sorted_fprs = [p[0] for p in sorted_pairs]
    sorted_tprs = [p[1] for p in sorted_pairs]
    auc = float(np.trapz(sorted_tprs, sorted_fprs))
    auc = max(0.0, min(1.0, abs(auc)))

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "roc_curve": {
            "fpr": np.round(sorted_fprs, 4).tolist(),
            "tpr": np.round(sorted_tprs, 4).tolist()
        }
    }
