import copy
import numpy as np
from cvxopt import matrix, solvers
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class LogReg:
    def __init__(self, lr=0.1, iters=3000):
        self.lr = lr
        self.iters = iters
        self.w = None
        self.b = 0.0
        self._sc = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        X, y = np.array(X, float), np.array(y, float)
        self._sc = StandardScaler()
        X = self._sc.fit_transform(X)
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        for _ in range(self.iters):
            p   = self._sigmoid(X @ self.w + self.b)
            err = p - y
            self.w -= self.lr * (X.T @ err) / m
            self.b -= self.lr * err.mean()
        return self

    def predict(self, X):
        X = np.array(X, float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = self._sc.transform(X)
        return (self._sigmoid(X @ self.w + self.b) > 0.5).astype(int)


class DecisionTree:
    def __init__(self, max_depth=5, min_samples=20):
        self.max_depth  = max_depth
        self.min_samples = min_samples
        self.tree = None

    def _entropy(self, y):
        if len(y) == 0: return 0
        _, c = np.unique(y, return_counts=True)
        p = c / c.sum()
        return -np.sum(p * np.log2(p + 1e-10))

    def _info_gain(self, col, y, t):
        left, right = y[col <= t], y[col > t]
        if len(left) == 0 or len(right) == 0: return 0
        n = len(y)
        return self._entropy(y) - (len(left)/n)*self._entropy(left) - (len(right)/n)*self._entropy(right)

    def _best_split(self, X, y):
        best_gain, best = 0.0, None
        for f in range(X.shape[1]):
            vals = np.unique(X[:, f])
            for i in range(len(vals) - 1):
                t = (vals[i] + vals[i+1]) / 2
                g = self._info_gain(X[:, f], y, t)
                if g > best_gain:
                    best_gain, best = g, (f, t)
        return best

    def _grow(self, X, y, depth=0):
        cls, cnt = np.unique(y, return_counts=True)
        majority = cls[np.argmax(cnt)]
        if len(cls) == 1 or depth >= self.max_depth or len(y) < self.min_samples:
            return {'leaf': majority}
        split = self._best_split(X, y)
        if split is None: return {'leaf': majority}
        f, t = split
        mask = X[:, f] <= t
        return {
            'feat': f, 'thresh': t,
            'left':  self._grow(X[mask],  y[mask],  depth + 1),
            'right': self._grow(X[~mask], y[~mask], depth + 1)
        }

    def fit(self, X, y):
        self.tree = self._grow(np.array(X, float), np.array(y))
        return self

    def _predict_one(self, x, node):
        if 'leaf' in node: return node['leaf']
        branch = 'left' if x[node['feat']] <= node['thresh'] else 'right'
        return self._predict_one(x, node[branch])

    def predict(self, X):
        X = np.array(X, float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self._predict_one(x, self.tree) for x in X])


class SVM:
    def __init__(self, C=1.0, kernel='linear', degree=2, gamma=0.1):
        self.C      = C
        self.kernel = kernel
        self.degree = degree
        self.gamma  = gamma
        self.alphas = None
        self.sv_X   = None
        self.sv_y   = None
        self.bias   = 0.0
        self._sc    = None

    def _k(self, a, b):
        if self.kernel == 'linear': return np.dot(a, b)
        if self.kernel == 'poly':   return (np.dot(a, b) + 1) ** self.degree
        if self.kernel == 'rbf':    return np.exp(-self.gamma * np.linalg.norm(a - b) ** 2)

    def fit(self, X, y):
        X, y = np.array(X, float), np.array(y)
        self._sc = StandardScaler()
        X = self._sc.fit_transform(X)
        y_pm = np.where(y == 0, -1, 1).astype(float)
        n = len(y_pm)

        K = np.array([[self._k(X[i], X[j]) for j in range(n)] for i in range(n)])

        P = matrix(np.outer(y_pm, y_pm) * K)
        q = matrix(-np.ones(n))
        G = matrix(np.vstack([-np.eye(n), np.eye(n)]))
        h = matrix(np.hstack([np.zeros(n), np.full(n, self.C)]))
        A = matrix(y_pm, (1, n))
        b = matrix(0.0)

        solvers.options['show_progress'] = False
        alphas = np.ravel(solvers.qp(P, q, G, h, A, b)['x'])

        sv = alphas > 1e-5
        self.alphas = alphas[sv]
        self.sv_X   = X[sv]
        self.sv_y   = y_pm[sv]

        self.bias = np.mean([
            self.sv_y[i] - sum(
                self.alphas * self.sv_y *
                np.array([self._k(self.sv_X[i], x) for x in self.sv_X])
            )
            for i in range(len(self.alphas))
        ])
        return self

    def predict(self, X):
        X = np.array(X, float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = self._sc.transform(X)
        scores = [
            sum(self.alphas * self.sv_y * np.array([self._k(x, sv) for sv in self.sv_X])) + self.bias
            for x in X
        ]
        return np.where(np.sign(scores) == -1, 0, 1).astype(int)


def majority_vote_cv(models, X, y, n_splits=10):
    """
    Evaluates majority voting ensemble via Stratified K-Fold cross validation.
    """
    kf     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for tr_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        preds   = np.array([copy.deepcopy(clf).fit(X_tr, y_tr).predict(X_val) for _, clf in models])
        majority = np.apply_along_axis(lambda col: np.bincount(col.astype(int)).argmax(), 0, preds)
        scores.append(accuracy_score(y_val, majority))
    return np.mean(scores)


def majority_vote_predict(models, X_tr, y_tr, X_te):
    """
    Fits all base models on full training data, and votes on predictions for the test set.
    """
    fitted  = [copy.deepcopy(clf).fit(X_tr, y_tr) for _, clf in models]
    preds   = np.array([clf.predict(X_te) for clf in fitted])
    return np.apply_along_axis(lambda col: np.bincount(col.astype(int)).argmax(), 0, preds)


def stacking(base_models, meta_model, X_tr, y_tr, X_te, n_splits=3):
    """
    Trains base models, generates Out-Of-Fold predictions, and trains the meta-learner on them.
    Returns meta-learner predictions on test set.
    """
    kf         = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    meta_tr    = np.zeros((len(y_tr), len(base_models)))
    fitted_bases = []

    for j, (name, clf) in enumerate(base_models):
        oof = np.zeros(len(y_tr))
        for tr_idx, val_idx in kf.split(X_tr, y_tr):
            m = copy.deepcopy(clf)
            m.fit(X_tr[tr_idx], y_tr[tr_idx])
            oof[val_idx] = m.predict(X_tr[val_idx])
        oof_acc = accuracy_score(y_tr, oof)
        print(f"  [{name}] OOF acc: {oof_acc:.4f}")
        meta_tr[:, j] = oof

        full = copy.deepcopy(clf)
        full.fit(X_tr, y_tr)
        fitted_bases.append(full)

    meta_te = np.column_stack([clf.predict(X_te) for clf in fitted_bases])
    meta    = copy.deepcopy(meta_model)
    meta.fit(meta_tr, y_tr)
    return meta.predict(meta_te)
