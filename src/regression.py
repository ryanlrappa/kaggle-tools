import pandas as pd 
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet, Lasso, RANSACRegressor, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer, median_absolute_error


class Regression:
    '''
    Class with methods for fitting, evaluating, and interpreting 
    useful regression models.

    Notes: 
        Consider features you may want to engineer,
        including polynomial and interaction terms 
        (based on EDA, initial regression results, etc.)
        before each instantiation of this class.

    Args:
        df: Pandas dataframe.
        X_cols (list of str): Feature column names.
        y_col (str): Target column name.

    Attributes:
        df: Pandas dataframe.
        X_cols (list of str): Feature column names.
        y_col (str): Target column name.
        X (np matrix): Features.
        y (np array): Target.
        X_scaled (np matrix): Standardized features.
        y_scaled (np array): Standardized target.
        y_log (np array): Log of target, generated if all target values are > 0.

    Todo:
        * add options for walk-forward CV
        * write Theil-Sen method

    '''

    def __init__(self, df, X_cols, y_col):
        self.X_cols = X_cols
        self.y_col = y_col
        self.df = df #pandas dataframe
        self.X = df[X_cols].values #getting features into numpy matrix for use with sklearn
        self.y = df[y_col].values.ravel() #getting target into numpy array for use with sklearn
        self.X_scaled = StandardScaler().fit(self.X).transform(self.X) #standardized feature matrix
        self.y_scaled = StandardScaler().fit(self.y.reshape(-1, 1)).transform(self.y.reshape(-1, 1)).ravel() #standardized target array
        if min(self.y) > 0:
            self.y_log = np.log10(self.y) #log of target, useful in case it's nonlinear
        else:
            print("Target includes negative values, log(y) attribute not generated.")


    def elastic_net(self, alphas, l1_ratios, folds, log_y=False): #slow (75 models -> 7 min)
        '''
        Runs parallelized GridSearchCV for elastic net regression
        over the specified alphas and l1_ratios.

        Args:
            alphas (list of floats >= 0): 0 is no penalty, max penalty is rarely > 10.
            l1_ratios (list of floats 0 to 1): 0 is pure ridge (l2 penalty), 1 is pure lasso (l1 penalty)
            folds (int): Number of folds to use in cross validation.
            log_y (bool): Default False, set to True to model on log(y) instead of y.

        Returns:
            model (obj): GridSearchCV optimized elastic net model.
            preds (np array): Predicted values of target.

        Raises:
            ~in progress~
        '''
        if log_y == False: #not using log(y)
            parameters = {'l1_ratio': l1_ratios, 'alpha': alphas}
            elastic_net = ElasticNet()
            scoring = make_scorer(mean_squared_error, greater_is_better=False) #so GridSearch settles on min MSE, not max R^2 which is default
            model = GridSearchCV(elastic_net, parameters, scoring=scoring, cv=folds, n_jobs=-1)
            model.fit(self.X_scaled, self.y)
            preds = model.predict(self.X_scaled)
            print("-------- BEST MODEL --------")
            print(model.best_estimator_)
            print("-------- ---------- --------")
            return model, preds
        else: #using log(y)
            parameters = {'l1_ratio': l1_ratios, 'alpha': alphas}
            elastic_net = ElasticNet()
            scoring = make_scorer(mean_squared_error, greater_is_better=False) #so GridSearch settles on min MSE, not max R^2 which is default
            model = GridSearchCV(elastic_net, parameters, scoring=scoring, cv=folds, n_jobs=-1)
            model.fit(self.X_scaled, self.y_log)
            preds = model.predict(self.X_scaled)
            print("-------- BEST MODEL --------")
            print(model.best_estimator_)
            print("-------- ---------- --------")
            return model, preds


    def elastic_net_sgd(self, alphas, l1_ratios, folds, log_y=False): #faster, more robust (75 models -> 30 sec)
        '''
        Runs parallelized GridSearchCV for SGD optimized elastic net regression
        over the specified alphas and l1_ratios.

        Notes:
            Predictions are more robust to outliers than plain elastic net,
            but appear to have more bias as well.
            Does not fit inliers as closely as RANSAC, Theil-Sen,
            so they may be preferable for robust predictions.

        Args:
            alphas (list of floats >= 0): 0 is no penalty, max penalty is rarely > 10.
            l1_ratios (list of floats 0 to 1): 0 is pure ridge (l2 penalty), 1 is pure lasso (l1 penalty)
            folds (int): Number of folds to use in cross validation.
            log_y (bool): Default False, set to True to model on log(y) instead of y.

        Returns:
            model (obj): GridSearchCV optimized elastic net model.
            preds (np array): Predicted values of target.

        Raises:
            ~in progress~
        '''
        if log_y == False: #not using log(y)
            parameters = {'l1_ratio': l1_ratios, 'alpha': alphas}
            elastic_net = SGDRegressor(penalty='elasticnet')
            scoring = make_scorer(mean_squared_error, greater_is_better=False) #so GridSearch settles on min MSE, not max R^2 which is default
            model = GridSearchCV(elastic_net, parameters, scoring=scoring, cv=folds, n_jobs=-1)
            model.fit(self.X_scaled, self.y)
            preds = model.predict(self.X_scaled)
            print("-------- BEST MODEL --------")
            print(model.best_estimator_)
            print("-------- ---------- --------")
            return model, preds
        else: #using log(y)
            parameters = {'l1_ratio': l1_ratios, 'alpha': alphas}
            elastic_net = SGDRegressor(penalty='elasticnet')
            scoring = make_scorer(mean_squared_error, greater_is_better=False) #so GridSearch settles on min MSE, not max R^2 which is default
            model = GridSearchCV(elastic_net, parameters, scoring=scoring, cv=folds, n_jobs=-1)
            model.fit(self.X_scaled, self.y_log)
            preds = model.predict(self.X_scaled)
            print("-------- BEST MODEL --------")
            print(model.best_estimator_)
            print("-------- ---------- --------")
            return model, preds


    def ransac(self, folds, log_y=False):
        '''For making predictions that are robust to outliers.

        Runs parallelized GridSearchCV on a RANSAC regressor
        over two possible values for min_samples.

        Args:
            folds (int): Number of folds to use in cross validation.

        Returns:
            model (obj): GridSearchCV optimized RANSAC model.
            preds (np array): Predicted values of target.

        '''
        if log_y == False: #not using log(y)
            #calculating min_samples (N) based on standard academic approach
            e = round(len(self.df[~(np.abs(stats.zscore(self.df)) < 3).all(axis=1)])/len(self.df), 4) #prob of outlier
            p = 0.99 #desired prob that at least one random sample is all inliers
            s = 2 #min num of points needed to fit model
            N = round(np.log10(1-p) / np.log10(1 - (1-e)**s))
            #comparing N to default setting for min_samples
            default = self.X.shape[1]+1
            parameters = {'min_samples': [default, N]}
            ransac = RANSACRegressor()
            scoring = make_scorer(median_absolute_error, greater_is_better=False)
            model = GridSearchCV(ransac, parameters, scoring=scoring, cv=folds, n_jobs=-1)
            model.fit(self.X_scaled, self.y)
            preds = model.predict(self.X_scaled)
            print("-------- BEST MODEL --------")
            print(model.best_estimator_)
            print("-------- ---------- --------")
            return model, preds
        else: #using log(y)
            #calculating min_samples (N) based on standard academic approach
            e = round(len(self.df[~(np.abs(stats.zscore(self.df)) < 3).all(axis=1)])/len(self.df), 4) #prob of outlier
            p = 0.99 #desired prob that at least one random sample is all inliers
            s = 2 #min num of points needed to fit model
            N = round(np.log10(1-p) / np.log10(1 - (1-e)**s))
            #comparing N to default setting for min_samples
            default = self.X.shape[1]+1
            parameters = {'min_samples': [default, N]}
            ransac = RANSACRegressor()
            scoring = make_scorer(median_absolute_error, greater_is_better=False)
            model = GridSearchCV(ransac, parameters, scoring=scoring, cv=folds, n_jobs=-1)
            model.fit(self.X_scaled, self.y_log)
            preds = model.predict(self.X_scaled)
            print("-------- BEST MODEL --------")
            print(model.best_estimator_)
            print("-------- ---------- --------")
            return model, preds


    def theil_sen(self):
        '''
        ~in progress~
        '''
        pass


    def coefficient_plot(self, model):
        '''Plots model coefficients.

        Args:
            model (obj): GridSearchCV model object.

        '''
        coef = pd.Series(model.best_estimator_.coef_, index=self.X_cols)
        sorted_coef = coef.sort_values()
        sorted_coef.plot(kind = "barh", figsize=(12, 9))
        plt.title("Coefficients in the model")
        print(sorted_coef[::-1])
        print("Intercept  ", model.best_estimator_.intercept_)


    def lasso_plot(self, alphas):
        '''Visualizes robust coefficients (feature importances).

        Args:
            alphas (list of floats >= 0): 0 is no penalty, max penalty is rarely > 10.

        Example:
            reg.lasso_plot(np.logspace(-5, 1, 25))

        '''
        coefs = []
        for a in alphas:
            lasso = Lasso(alpha=a)
            lasso.fit(self.X_scaled, self.y)
            coefs.append(lasso.coef_)
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        ax.plot(alphas, coefs)
        ax.set_xscale('log')
        # ax.set_xlim(ax.get_xlim()[::-1]) #reverse axis
        plt.xlabel('alpha (λ)')
        plt.ylabel('coefficients')
        plt.title('LASSO coefficients as a function of regularization')
        plt.legend(labels=self.X_cols, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()


    def plot_residuals(self, preds):
        '''
        Plots predicted values against their residuals.
        Also prints MSE, MAE, R^2, and Median Absolute Error.

        Notes:
            http://docs.statwing.com/interpreting-residual-plots-to-improve-your-regression/

        Args:
            preds (list or np array): Predicted values of the target.

        '''
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.scatter(preds, preds - self.y) #NOTE: may want to plot standardized residuals instead
        ax.set_xlabel("Predicted values: " + self.y_col)#, fontsize=15)
        ax.set_ylabel("Residuals")#, fontsize=15)
        plt.axhline(y=0, color='k', linestyle='--')
        plt.show()
        print("MSE:", mean_squared_error(self.y, preds))
        print("MAE:", mean_absolute_error(self.y, preds))
        print("R^2:", r2_score(self.y, preds))
        print("Median Absolute Error:", median_absolute_error(self.y, preds))
