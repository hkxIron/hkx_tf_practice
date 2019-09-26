from mlxtend.regressor import StackingCVRegressor
from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np

RANDOM_SEED = 42
X, y = load_boston(return_X_y=True)

ridge = Ridge()
lasso = Lasso()
rf = RandomForestRegressor(random_state=RANDOM_SEED)
# The StackingCVRegressor uses scikit-learn's check_cv
# internally, which doesn't support a random seed. Thus
# NumPy's random seed need to be specified explicitely for
# deterministic behavior
np.random.seed(RANDOM_SEED)

stack = StackingCVRegressor(regressors=(lasso, ridge),
                            meta_regressor=rf,
                            use_features_in_secondary=True)
params = {'lasso__alpha': [0.1, 1.0, 10.0],
          'ridge__alpha': [0.1, 1.0, 10.0]}

grid = GridSearchCV(estimator=stack,
                    param_grid={
                        'lasso__alpha': [x/5.0 for x in range(1, 10)],
                        'ridge__alpha': [x/20.0 for x in range(1, 10)],
                        'meta-randomforestregressor__n_estimators': [10,100]
                    },
                    cv=5,
                    refit=True
)

grid.fit(X, y)

print("Best: %f using %s" % (grid.best_score_, grid.best_params_))

#Best: 0.673590 using {'lasso__alpha': 0.4, 'meta-randomforestregressor__n_estimators': 10, 'ridge__alpha': 0.3

cv_keys = ('mean_test_score', 'std_test_score', 'params')
for r, _ in enumerate(grid.cv_results_['mean_test_score']):
  print("%0.3f +/- %0.2f %r"
          % (grid.cv_results_[cv_keys[0]][r],
          grid.cv_results_[cv_keys[1]][r] / 2.0,
          grid.cv_results_[cv_keys[2]][r]))
  if r > 10:
    break
print('...')

print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %.2f' % grid.best_score_)