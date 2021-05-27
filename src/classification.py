import pandas as pd 
import numpy as np
import itertools
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, roc_auc_score, average_precision_score, classification_report, roc_curve, precision_recall_curve
from sklearn.linear_model import LogisticRegression, SGDClassifier


class Classification:
    '''
    Class with methods for fitting, evaluating, and interpreting 
    useful classification models.

    Note: 
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

    Todo:
        * GB
        * add options for walk-forward CV
        * rewrite class/init to do an inital train-test split?

    '''

    def __init__(self, df, X_cols, y_col):
        self.X_cols = X_cols
        self.y_col = y_col
        self.df = df #pandas dataframe
        self.X = df[X_cols].values #getting features into numpy matrix for use with sklearn
        self.y = df[y_col].values.ravel() #getting target into numpy array for use with sklearn
        self.X_scaled = StandardScaler().fit(self.X).transform(self.X) #standardized feature matrix


    def elastic_net(self, alphas, l1_ratios, folds, imbalanced=False, time_series=False): #slow
        # NOTE: Should I consider class_weight? Does
        # class_weight just rescale the predicted probabilities,
        # or change result? And does it imply assumptions about
        # cost/benefit (trades off recall for precision)?
        '''
        Runs parallelized RandomizedSearchCV for elastic net logistic
        regression over the specified alphas and l1_ratios.

        Notes:
            Instead of alphas for regularization strength, 
            sklearn logistic reg uses parameter 'C' which is inverse alphas.

        Args:
            alphas (list of floats > 0): +infinity approaches no penalty, max penaly is rarely < 0.1
            l1_ratios (list of floats 0 to 1): 0 is pure ridge (l2 penalty), 1 is pure lasso (l1 penalty)
            folds (int): Number of folds to use in cross validation.
            imbalanced (bool): Default False, set to True to score on PR instead of ROC.
            time_series (bool): Default False, set to True to use expanding window walk-forward CV.

        Returns:
            model (obj): RandomizedSearchCV optimized elastic net model.
            preds (np array): Predicted values of target.

        Raises:
            # TODO

        Example:
            alphas = np.logspace(1, -2, 10)
            l1_ratios = [0, .5, 1]
            folds = 5
            model, preds = clf.elastic_net(alphas, l1_ratios, folds)

        '''
        if time_series == False:
            if imbalanced == False:
                scoring = make_scorer(roc_auc_score, needs_threshold=True)
            else:
                scoring = make_scorer(average_precision_score, needs_threshold=True)
            parameters = {'l1_ratio': l1_ratios, 'C': alphas}
            elastic_net = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=1000, n_jobs=-1)
            scoring = make_scorer(average_precision_score, needs_threshold=True)
            model = RandomizedSearchCV(elastic_net, parameters, scoring=scoring, cv=folds, n_jobs=-1)
            model.fit(self.X_scaled, self.y)
            preds = model.predict(self.X_scaled)
            print("-------- BEST MODEL --------")
            print(model.best_estimator_)
            print("-------- ---------- --------")
            return model, preds
        else:
            if imbalanced == False:
                scoring = make_scorer(roc_auc_score, needs_threshold=True)
            else:
                scoring = make_scorer(average_precision_score, needs_threshold=True)
            parameters = {'l1_ratio': l1_ratios, 'C': alphas}
            elastic_net = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=1000, n_jobs=-1)
            tscv = TimeSeriesSplit(n_splits=folds)
            model = RandomizedSearchCV(elastic_net, parameters, scoring=scoring, cv=tscv, n_jobs=-1)
            model.fit(self.X_scaled, self.y)
            preds = model.predict(self.X_scaled)
            print("-------- BEST MODEL --------")
            print(model.best_estimator_)
            print("-------- ---------- --------")
            return model, preds


    def elastic_net_sgd(self, alphas, l1_ratios, folds, time_series=False): #faster, more robust
        '''
        Runs parallelized RandomizedSearchCV for SGD optimized elastic net 
        logistic regression over the specified alphas and l1_ratios.
        Also optimizes over class_weight 'balanced' vs None.

        Notes:
            Predictions are more robust to outliers than plain elastic net,
            but appear to have more bias as well.
            Does not fit inliers as closely as RF or GB
            so they may be preferable for robust predictions.

        Args:
            alphas (list of floats >= 0): 0 is no penalty, max penalty is rarely > 10.
            l1_ratios (list of floats 0 to 1): 0 is pure ridge (l2 penalty), 1 is pure lasso (l1 penalty)
            folds (int): Number of folds to use in cross validation.
            time_series (bool): Default False, set to True to use expanding window walk-forward CV.

        Returns:
            model (obj): RandomizedSearchCV optimized elastic net model.
            preds (np array): Predicted values of target.

        Raises:
            ~in progress~

        Example:
            alphas = np.logspace(-3, 1, 25)
            l1_ratios = [0, .5, 1]
            folds = 5
            model, preds = clf.elastic_net_sgd(alphas, l1_ratios, folds)
        
        '''
        parameters = {'l1_ratio': l1_ratios, 'alpha': alphas, 'class_weight': [None, 'balanced']}
        elastic_net = SGDClassifier(loss='log', penalty='elasticnet', n_jobs=-1)
        if time_series == False:
            model = RandomizedSearchCV(elastic_net, parameters, cv=folds, n_jobs=-1)
        else:
            tscv = TimeSeriesSplit(n_splits=folds)
            model = RandomizedSearchCV(elastic_net, parameters, cv=tscv, n_jobs=-1)
        model.fit(self.X_scaled, self.y)
        preds = model.predict(self.X_scaled)
        print("-------- BEST MODEL --------")
        print(model.best_estimator_)
        print("-------- ---------- --------")
        return model, preds


    def coefficient_plot(self, model):
        '''Plots model coefficients.

        Args:
            model (obj): RandomizedSearchCV model object.

        '''
        if len(model.best_estimator_.coef_) == 1: #this happens with Logistic Reg coefs
            coef = pd.Series(model.best_estimator_.coef_[0], index=self.X_cols)
        else:
            coef = pd.Series(model.best_estimator_.coef_, index=self.X_cols)
        sorted_coef = coef.sort_values()
        sorted_coef.plot(kind = "barh", figsize=(12, 9))
        plt.title("Coefficients in the model")
        print(sorted_coef[::-1])
        print("Intercept  ", model.best_estimator_.intercept_)


    def lasso_plot(self, alphas):
        '''Visualizes robust coefficients (feature importances).

        Args:
            alphas (list of floats > 0): +infinity approaches no penalty, max penaly is rarely < 0.1

        Example:
            clf.lasso_plot(np.logspace(2, -3, 25))

        '''
        coefs = []
        for a in alphas:
            lasso = LogisticRegression(penalty='l1', C=a, solver='saga', max_iter=1000)
            lasso.fit(self.X_scaled, self.y)
            if len(lasso.coef_) == 1:
                coefs.append(lasso.coef_[0])
            else:
                coefs.append(lasso.coef_)
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        ax.plot(alphas, coefs)
        ax.set_xscale('log')
        # ax.set_xlim(ax.get_xlim()[::-1]) #reverse axis
        plt.xlabel('C (inverse λ)')
        plt.ylabel('coefficients')
        plt.title('LASSO coefficients as a function of regularization')
        plt.legend(labels=self.X_cols, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()


    def roc_curve(self, model):
        '''Plots FPR against TPR across decision thresholds.
        
        Args:
            model (obj): Sklearn classifier.

        '''
        probs = model.predict_proba(self.X_scaled) #predict probabilities
        probs = probs[:, 1] #keep probabilities for the positive outcome only
        fpr, tpr, thresholds = roc_curve(self.y, probs) #calculate roc curve
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], linestyle='--') #plot no skill
        plt.plot(fpr, tpr) #plot the roc curve for the model
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.show()

        best_ratio = 0
        best_thresh = 0
        corresp_tp = 0
        corresp_fp = 0
        for fp, tp, t in zip(fpr, tpr, thresholds):
            if tp/fp > best_ratio and tp > 0.5:
                best_ratio = tp/fp
                best_thresh = t
                corresp_tp = tp
                corresp_fp = fp
        print("Best ratio w/ TPR > .5:", best_ratio)
        print("    Decision threshold:", best_thresh)
        print("                   TPR:", corresp_tp)
        print("                   FPR:", corresp_fp)


    def pr_curve(self, model):
        '''Plots Recall (TPR) against Precision across decision thresholds.
        
        Args:
            model (obj): Sklearn classifier.

        '''
        probs = model.predict_proba(self.X_scaled) #predict probabilities
        probs = probs[:, 1] #keep probabilities for the positive outcome only
        precision, recall, thresholds = precision_recall_curve(self.y, probs) #calculate p-r curve
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0.5, 0.5], linestyle='--') #plot no skill
        plt.plot(recall, precision) #plot the p-r curve for the model
        plt.xlabel('Recall (TPR)')
        plt.ylabel('Precision')
        plt.show()

        best_ratio = 0
        best_thresh = 0
        corresp_p = 0
        corresp_r = 0
        for p, r, t in zip(precision, recall, thresholds):
            if p/r < 1 and p/r > best_ratio:
                best_ratio = p/r
                best_thresh = t
                corresp_p = p
                corresp_r = r
        print("P/R ratio closest to 1:", best_ratio)
        print("    Decision threshold:", best_thresh)
        print("             Precision:", corresp_p)
        print("                Recall:", corresp_r)


    def profit_curve(self, thresholds, model, cost_bene):
        '''
        Given assumptions about cost/benefit of TPs, FPs TNs, FNs, 
        plots profit of classifier across decision thresholds
        and prints threshold for max profit.

        Args:
            thresholds (arr of floats): List of decision thresholds to plot.
            model (obj): Sklearn classifier.
            cost_bene (dict): Dict mapping costs/benefits to TP, FP, TN, FN.

        Example:
            cost_bene = {'tp': 2, 'fp': -2, 'tn': 0, 'fn': -1}
            model = SGDClassifier()
            thresholds = np.linspace(.01, .99, 25)
            clf.profit_curve(thresholds, model, cost_bene)

        '''
        profits = []
        for t in thresholds:
            preds = (model.predict_proba(self.X_scaled)[:,1] >= t).astype(bool)  
            y = self.y
            tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
            avg_profit = (tn*cost_bene['tn'] + fp*cost_bene['fp'] + fn*cost_bene['fn'] + tp*cost_bene['tp'])/(tn+fp+fn+tp)
            profits.append(avg_profit)

        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, profits)
        plt.xlabel('Decision threshold')
        plt.ylabel('Expected profit')
        plt.show()
        
        print("Estimated max avg profit:", max(profits))
        print("      Decision threshold:", thresholds[profits.index(max(profits))])



class ConfusionMatrix:
    '''
    Class for plotting confusion matrix and 
    printing accuracy, precision, recall, and fallout.

    Args:
        y (list or np array): Actual target values.
        y_pred (list or np array): Predicted target values.
        model (obj): Sklearn classifier.

    Example:
        cm = ConfusionMatrix(y_pred, y_test, model)
        cm.plot_matrix()

    '''
    
    def __init__(self, y, y_pred, model):
        self.y = y #e.g. df[y_col]
        self.y_pred = y_pred
        self.model = model
        self.cm = confusion_matrix(self.y, self.y_pred)
        if model.classes_[0] == 1: #in case the labels are flipped from the usual indices
            self.cm = np.array([[self.cm[1,1], self.cm[1,0]], [self.cm[0,1], self.cm[0,0]]])


    def plot_matrix(self, classes=['0', '1'], title_on=False, title='Confusion Matrix', cmap=plt.cm.Blues, **kwargs):
        '''Plots confusion matrix w/ accuracy, precision, recall, fallout.
        
        Args:
            classes (list of two str): ['0', '1'] by default, change labels as desired.
            title_on (bool, optional): Default False, set to True to print title.
            title (str): Show title, if title_on arg set to True.
            cmap: Plot color palette.
            **kwargs: For use with sklearn classification report below.

        '''

        fig, ax = plt.subplots()
        
        plt.imshow(self.cm, interpolation='nearest', cmap=cmap)
        if title_on == True:
            plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = self.cm.max() / 2.
        for i, j in itertools.product(range(self.cm.shape[0]), range(self.cm.shape[1])):
            plt.text(j, i, self.cm[i, j],
                    horizontalalignment="center",
                    color="white" if self.cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label', fontsize=14)
        ax.xaxis.tick_top()
        plt.xlabel('Predicted label', fontsize=14)
        ax.xaxis.set_label_position('top')
        plt.show()

        #for comparison to the classification report
        tp = self.cm[1,1]
        fn = self.cm[1,0]
        fp = self.cm[0,1]
        tn = self.cm[0,0]
        print('Accuracy =      {:.3f}'.format((tp+tn)/(tp+fp+tn+fn)))
        print('Precision =     {:.3f}'.format(tp/(tp+fp)))
        print('Recall (TPR) =  {:.3f}'.format(tp/(tp+fn)))
        print('Fallout (FPR) = {:.3f}'.format(fp/(fp+tn)))

        #classification report
        print("")
        print('---- Classification Report ----')
        print(classification_report(self.y, self.y_pred, **kwargs))
