import joblib
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import warnings 
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

class Ml_models_regression:

    def __init__(self, train_features, train_labels, metrics):
        self.train_features = train_features
        self.train_labels = train_labels
        self.metrics = metrics 

    def linear_regression(self):
        lr = LinearRegression()
        lr.fit(self.train_features, self.train_labels.values.ravel())
        joblib.dump(lr, "../tmp_models/LR_model.pkl")
        
    def ridge_regression(self):
        ridge = Ridge()
        parameters = {
            'alpha' : [0.001, 0.01, 0.1, 1, 10]
        }
        cv = GridSearchCV(ridge, parameters, cv=5, verbose=10, scoring=self.metrics)
        cv.fit(self.train_features, self.train_labels.values.ravel())
        joblib.dump(cv.best_estimator_, "../tmp_models/ridge_model.pkl")
        self.print_results(cv)
    
    def elasticnet_regression(self):
        enet = ElasticNet()
        parameters = {
            'alpha' : [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'l1_ratio' : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        }
        cv = GridSearchCV(enet, parameters, cv=5, verbose=10, scoring=self.metrics)
        cv.fit(self.train_features, self.train_labels.values.ravel())
        joblib.dump(cv.best_estimator_, "../tmp_models/elasticnet_model.pkl")
        self.print_results(cv)
     
    def lasso_regression(self):
        lasso = Lasso()
        parameters = {
            'alpha' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }
        cv = GridSearchCV(lasso, parameters, cv=5, verbose=10, scoring=self.metrics)
        cv.fit(self.train_features, self.train_labels.values.ravel())
        joblib.dump(cv.best_estimator_, "../tmp_models/lasso_model.pkl")
        self.print_results(cv)
        
    def svm_regressor(self):
        svr = SVR()
        parameters = {
            'C': [0.01, 0.1, 1, 10, 100],
            'kernel':['linear','rbf']
        }
        cv = GridSearchCV(svr, parameters, cv=5, verbose=10, scoring=self.metrics)
        cv.fit(self.train_features, self.train_labels.values.ravel())
        joblib.dump(cv.best_estimator_, "../tmp_models/svm_model.pkl")
        self.print_results(cv)
        
    def decision_trees_regressor(self):
        dtr = DecisionTreeRegressor()
        parameters = {
            'max_depth':[16, 32, 40, 50, 100, None]
        }
        cv = GridSearchCV(dtr, parameters, cv=5, verbose=10, scoring=self.metrics)
        cv.fit(self.train_features, self.train_labels.values.ravel())
        joblib.dump(cv.best_estimator_, "../tmp_models/dtr_model.pkl")
        self.print_results(cv)
        
    def polynomial_regression(self):
        degree = 3 
        polyreg=make_pipeline(PolynomialFeatures(degree),LinearRegression())
        polyreg.fit(self.train_features, self.train_labels.values.ravel())
        joblib.dump(polyreg, "../tmp_models/poly_model.pkl")
        
    def gradient_boosting_regressor(self):
        gbr = GradientBoostingRegressor()
        parameters = {
            'n_estimators':[100, 200],
            'max_depth':[16, 128, None],
            'learning_rate':[0.01, 0.1, 1, 10]
        }
        cv = GridSearchCV(gbr, parameters, cv=5, verbose=10, scoring=self.metrics)
        cv.fit(self.train_features, self.train_labels.values.ravel())
        joblib.dump(cv.best_estimator_, "../tmp_models/gbr_model.pkl")
        self.print_results(cv)
        
    def xgboosting_regressor(self):
        xgbr = XGBRegressor(objective='reg:squarederror')
        parameters = {
            'max_depth':[16, 32, None]
        }
        cv = GridSearchCV(xgbr, parameters, cv=5, verbose=10, scoring=self.metrics)
        cv.fit(self.train_features, self.train_labels.values.ravel())
        joblib.dump(cv.best_estimator_, "../tmp_models/xgbr_model.pkl")
        self.print_results(cv)
        
    def random_forest_regressor(self):
        rfr = RandomForestRegressor()
        parameters = {
            'n_estimators':[5,100,200],
            'max_depth':[16, 32, None]
        }
        cv = GridSearchCV(rfr, parameters, cv=5, verbose=10, scoring=self.metrics)
        cv.fit(self.train_features, self.train_labels.values.ravel())
        joblib.dump(cv.best_estimator_, "../tmp_models/rfr_model.pkl")
        self.print_results(cv)
        
    def print_results(self, results):
        print("BEST Parameters: {}\n".format(results.best_params_))
        means = results.cv_results_['mean_test_score']
        stds = results.cv_results_['std_test_score']

        for mean, std, params in zip(means, stds, results.cv_results_['params']):
            print('{} (+/-{}) for {}'.format(round(mean,3), round(std * 2,3), params))
