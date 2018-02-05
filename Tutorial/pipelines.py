from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer

my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())

my_pipeline.fit(train_X, train_y)
predictions = my_pipeline.predict(test_X)