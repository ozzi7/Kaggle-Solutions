import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np
from xgboost import XGBClassifier


def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

rmsle_score = make_scorer(rmsle, greater_is_better=False)

santander_train_file_path = 'train.csv'
santander_test_file_path = 'test.csv'

santander_data = pd.read_csv(santander_train_file_path)
santander_test_data = pd.read_csv(santander_test_file_path)

X_test = santander_test_data.drop(['ID'], axis=1)

y = np.log1p(santander_data.target).values
X = santander_data.drop(['ID','target'], axis=1)

# Define model
santander_gradient_boosting_reg = GradientBoostingRegressor(learning_rate=0.0001, n_estimators=50,
                                                            max_features='log2', min_samples_split=2, max_depth=1,
                                                            verbose=2)
santander_RF = RandomForestRegressor(random_state=1762, n_jobs=2, verbose=2, criterion='mse')
santander_XGBClassifier = XGBClassifier()

# To evaluate the model
kfold = KFold(n_splits=3, random_state=347)
scores = cross_val_score(santander_RF, X, y, cv=kfold, scoring=rmsle_score)

# Fit model to all testing data (not just part of it)
santander_RF.fit(X, y)

# Predict submission
predicted = santander_RF.predict(X_test)
predicted_exp = np.expm1(predicted)
submission = pd.DataFrame({'ID': santander_test_data.ID, 'target': predicted_exp})
submission.to_csv('submission.csv', index=False)