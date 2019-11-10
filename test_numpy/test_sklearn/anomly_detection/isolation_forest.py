import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy import stats



rng = np.random.RandomState(42)

n_samples=10  #样本总数
data = np.random.rand(n_samples,3)
data[8,:] -= 5
data[9,:] += 3
df = pd.DataFrame(data)


# fit the model
clf = IsolationForest(max_samples=n_samples, random_state=rng, contamination=0.2)  #contamination为异常样本比例
clf.fit(df.values)
scores_pred = clf.decision_function(df.values)
print(scores_pred)
print(len(scores_pred))
outliers_fraction = 0.5
threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)

print("pred label:", clf.predict(df.values))
