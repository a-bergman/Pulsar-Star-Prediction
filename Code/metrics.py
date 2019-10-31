# Imports

import pandas        as pd
from sklearn.metrics import confusion_matrix, r2_score, balanced_accuracy_score
from sklearn.metrics import recall_score, f1_score, roc_auc_score, matthews_corrcoef

"""
The docstrings for each graph contain the following:

- parameters  : values which must be entered, some of which have defaults
- description : what each function does
- returns     : the output of each function

The parameters section of each docstring is set up as:

parameter : definition : type : possible values (if applicable)

These functions are designed to build off of what is available in already sci-kit learn: 
either to add a metric that does not exist or to improve something does already exist.
"""

"""class RegressionMetrics():

    def __init__(self, name):
        self.name = name

    if __name__ == "__main__":
        main()
"""


# Regression Metrics

def r2_adj(X, y, y_predicted):
    """
    Parameters:
    -----------
        X           : the X variables   :
        y           : the true values   :
        y_predicted : model predictions :

    Description:
    ------------
    Calculates an adjusted R^2 score which is scaled to the number of features in the model: the R^2 score is often inflated by a large number of features.

    Returns:
    --------
    The coefficient of correlation: a floating point number between 0 and 1.
    """
    r2 = r2_score(y_true, y_predicted)
    numerator = (1 - r2) * (len(y) - 1)
    denominator = (len(y) - len(X.columns)) - 1
    quotient = numerator / denominator
    r2_adj = 1 - quotient
    return r2_adj

# Classification Metrics

def confusion_matrix_dataframe(y, y_predicted, columns, index):
    """
    Parameters:
    -----------
    y           : the true values       :     :
    y_predicted : the model predictions :     :
    columns     : column labels         : str : [0, 1, etc.]
    index       : row labels            : str : [0, 1, etc.]
    
    Description:
    ------------
    Generates a confusion matrix through sklearn and transforms it into a Pandas dataframe.
    This can work with binary or multi-class classification.

    Returns:
    --------
    A Pandas dataframe of the sklearn's confusion_matrix.
    """
    cm     = confusion_matrix(y, y_predicted)
    matrix = pd.DataFrame(cm, columns = columns, index = index)
    return matrix

def specificity(y, y_predicted):
    """
    Parameters:
    -----------
    y           : the true values       : :
    y_predicted : the model predictions : :

    Description:
    ------------
    Calculates the percentage of negatives that are correctly classified as being negative. A confusion matrix generated and is the score (TN / TN + FP) is calculated.

    Returns:
    --------
    The specificity score: a floating point number between 0 and 1
    """
    cm = confusion_matrix(y, y_predicted)  
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])
    return specificity

def ternary_specificity(y, y_predicted):
    """
    Parameters:
    -----------
    y           : the true values       : :
    y_predicted : the model predictions : :

    Description:
    ------------
    Calculates the percentage of "negative" classes that are classified correctly as "negative".  A confusion matrix is generated and the scores for each class are
    averaged.

    Returns:
    --------
    The specificity score: a floating point number between 0 and 1.
    """
    cm = confusion_matrix(y, y_predicted)
    s1 = cm[0,0] / (cm[0,0] + cm[0,1] + cm[0,2])
    s2 = cm[1,1] / (cm[0,1] + cm[1,1] + cm[1,2])
    s3 = cm[2,2] / (cm[0,2] + cm[1,2] + cm[2,2])
    specificity = (s1 + s2 + s3) / 3
    return specificity

def ternary_classification_summary(y, y_predicted):
    bal_acc = balanced_accuracy_score(y, y_predicted)
    recall  = recall_score(y, y_predicted, average = "macro")
    spec    = ternary_specificity(y, y_predicted)
    mcc     = matthews_corrcoef(y, y_predicted)
    classification_summary = pd.DataFrame([bal_acc, recall, spec, mcc], index = ["Balanced Accuracy", "Sensitivity", "Specificity", "Matthew Corr. Coef."]).T
    return classification_summary