import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
plt.rcParams.update({'figure.max_open_warning': 200})

class EDA:
    '''
    Class for performing basic exploratory data anaysis,
    given a pandas dataframe containing a numerical target (y)
    and numerical features (X)

    Args:
        df: pandas dataframe.
        X_cols (list of str): Feature column names.
        y_col (str): Target column name.

    Attributes:
        df: pandas dataframe.
        X_cols (list of str): Feature column names.
        y_col (str): Target column name.
    
    Todo:
        * countplots (for classification problems):
            - see: https://www.kaggle.com/stephaniestallworth/titanic-eda-classification-end-to-end
            - https://seaborn.pydata.org/generated/seaborn.countplot.html
            - https://stackoverflow.com/questions/31749448/how-to-add-percentages-on-top-of-bars-in-seaborn
            - also see: https://towardsdatascience.com/doing-eda-on-a-classification-project-pandas-crosstab-will-change-your-life-c61c1cb2c20b
        * plots for time series

    '''
    
    def __init__(self, df, X_cols, y_col):
        self.df = df
        self.X_cols = X_cols
        self.y_col = y_col


    def boxplots(self, hue=False): #fast
        '''Shows boxplots for every column in the dataframe.

        Args:
            hue (str, optional): Default False, or specify col name.

        '''
        if hue == False:
            for i, col in enumerate(self.df.columns):
                plt.figure(i)
                sns.boxplot(x=self.df[col])
        else:
            for i, col in enumerate(self.df.columns):
                plt.figure(i)
                sns.boxplot(x=self.df[hue], y=self.df[col])


    def histograms(self, hue=False, **kwargs): #fast
        '''Shows histograms for every column in the dataframe.

        Note:
            When using the hue arg, make sure target col is bool.
        
        Args:
            hue (str, optional): Default False.
            all kwargs: https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.hist.html

        Example:
            kwargs = {'density':True} #<-- good for imbalanced classification problems
            eda.histograms(hue=True, **kwargs)

        '''
        if hue == False:    
            for i, col in enumerate(self.df.columns):
                plt.figure(i)
                x=self.df[col]
                sns.distplot(x)
        else:
            for i, col in zip(range(len(self.df)), self.df.columns):
                if self.df.iloc[:, i].dtype in ('float64', 'int64'):
                    plt.figure()
                    plt.hist(self.df[self.df[self.y_col] == 0].iloc[:, i], bins=50, label='Fail', alpha=0.5, edgecolor='black', color='grey', **kwargs)
                    plt.hist(self.df[self.df[self.y_col] == 1].iloc[:, i], bins=50, label='Success', alpha=0.5, edgecolor='black', color='lightgreen', **kwargs)
                    plt.xlabel(col)
                    plt.ylabel(self.y_col)
                    plt.legend()
                else:
                    continue


    def jointplots(self): #fast-ish
        '''Scatters each feature col against target w/ axial histograms.'''
        for i, col in enumerate(self.X_cols):
            plt.figure(i)
            sns.jointplot(x=self.df[col], y=self.df[self.y_col], kind='scatter')
            print("Pearson's r for " + self.y_col + " & " + col + ":", self.df[col].corr(self.df[self.y_col]))


    def scatterplots_sns(self, **kwargs): #speed depends upon kwargs
        '''
        Scatters feature columns against target w/ several optional kwargs.

        Args:
            fit_reg (bool, optional): Default True, set to False to get rid of OLS best fit lines (very slow if True).
            robust (bool, optional): Default False, set to True to see best fit lines robust to outliers (extremely slow).
            hue (str, optional): Column name, colors datapoints by category from the given column (fast-ish).
            scatter_kws={'alpha':0.X} (optional): Make points more transparent. Lower X -> more transparency.
            all kwargs: https://seaborn.pydata.org/generated/seaborn.lmplot.html

        Example:
            kwargs = {'fit_reg':False, 'hue':'col_name', 'scatter_kws':{'alpha':0.3}}
            obj.scatterplots_sns(**kwargs)
        
        '''
        for i, col in enumerate(self.X_cols):
            plt.figure(i)
            sns.lmplot(x=col, y=self.y_col, data=self.df, **kwargs)
            print("Pearson's r for " + self.y_col + " & " + col + ":", self.df[col].corr(self.df[self.y_col]))



#Note on point transparency in seaborn plots:
#Use kwarg scatter_kws={'alpha':0.3}

#Note on ID'ing and removing outliers.
#Do I want to make a function for this?
# from scipy import stats
# len(df[~(np.abs(stats.zscore(df)) < 3).all(axis=1)]) #number of rows w/ outliers
# round(len(df[~(np.abs(stats.zscore(df)) < 3).all(axis=1)])/len(df), 4) #pct of rows w/ outliers
# df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)] #keep only rows w/o outliers