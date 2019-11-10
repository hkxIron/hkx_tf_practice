import numpy as np
from lightgbm_cv import LGBMClassifierCV
from optimizer import Optimizer


class LGBMOptimizer(Optimizer):
    """https://www.jianshu.com/p/1100e333fcab"""

    def __init__(self, X, y, cv=5, workers=5, random_state=None, params_bounds=None):
        super().__init__(X, y, cv, workers, random_state, params_bounds)
        self.params_bounds['n_estimators'] = 3000

    def objective(self, **params):
        """重写目标函数"""

        # 纠正参数类型
        for p in ('max_depth', 'depth', 'num_leaves', 'subsample_freq', 'min_child_samples'):
            if p in params:
                params[p] = int(np.round(params[p]))
        _params = {**self.params_bounds, **params}

        # 核心逻辑
        self.clf = LGBMClassifierCV(_params, self.cv, self.random_state)
        return self.clf.fit(self.X, self.y, early_stopping_rounds=300, verbose=0)

if __name__ == '__main__':
    from sklearn.datasets import make_classification

    X, y = make_classification(10000)
    opt = LGBMOptimizer(X, y)
    print(opt.maximize(1))
