from scipy import stats as sp_stats
from scipy import special
import numpy as np

class IdentityTransform:
    def fit(self, y):
        return self

    def transform(self, y):
        return y.copy()

    def inverse_transform(self, y):
        return y.copy()

class Log1pTransform:
    def fit(self, y):
        self.shift = 0.0
        if np.min(y) <= 0:
            self.shift = -float(np.min(y)) + 1.0
        return self

    def transform(self, y):
        return np.log1p(y + self.shift)

    def inverse_transform(self, y):
        return np.expm1(y) - self.shift

class BoxCoxTransform:
    def fit(self, y):
        arr = y.copy().astype(float)
        self.shift = 0.0
        if np.min(arr) <= 0:
            self.shift = -float(np.min(arr)) + 1.0
            arr += self.shift
        _, lam = sp_stats.boxcox(arr)
        self.lmbd = float(np.clip(lam, -2.0, 2.0))
        transformed = sp_stats.boxcox(arr, self.lmbd)
        self.bc_min = float(np.min(transformed))
        self.bc_max = float(np.max(transformed))
        self.orig_max = float(np.max(y))
        return self

    def transform(self, y):
        return sp_stats.boxcox(y.astype(float) + self.shift, self.lmbd)

    def inverse_transform(self, y):
        bc_range = max(abs(self.bc_max - self.bc_min), 1e-6)
        left = self.bc_min - 3 * bc_range
        right = self.bc_max + 3 * bc_range
        if self.lmbd > 0:
            left = max(left, -1.0 / self.lmbd + 1e-6)
        elif self.lmbd < 0:
            right = min(right, -1.0 / self.lmbd - 1e-6)
        y_safe = np.clip(y, left, right)
        res = special.inv_boxcox(y_safe, self.lmbd) - self.shift
        res = np.where(np.isfinite(res), res, self.orig_max)
        return np.clip(res, 0, self.orig_max * 5)

class DifferencingTransform:
    def fit(self, y):
        self.first_val = float(y[0])
        return self

    def transform(self, y):
        return np.diff(y)

    def inverse_transform(self, y_diff):
        return np.concatenate([[self.first_val], self.first_val + np.cumsum(y_diff)])

    def inverse_transform_forecast(self, y_diff, last_known):
        return last_known + np.cumsum(y_diff)


def get_transform(name: str):
    mapping = {
        "identity": IdentityTransform,
        "log1p": Log1pTransform,
        "boxcox": BoxCoxTransform,
        "differencing": DifferencingTransform,
    }
    return mapping[name]()
