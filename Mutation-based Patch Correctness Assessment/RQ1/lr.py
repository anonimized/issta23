import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from numpy import mean


def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]


def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]


def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]


def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]


def auc_(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    return auc(fpr, tpr)


if __name__ == '__main__':
    df = pd.read_csv('../Measurements/dataset.csv', delim_whitespace=False)
    df['Label_n'] = 1 - (LabelEncoder()).fit_transform(y=df['Label'])
    X = df.drop(labels=['Label', 'Id', 'Label_n'], axis='columns')
    y = df['Label_n']

    nm = NearMiss()
    X, y = nm.fit_resample(X, y)

    # create dataset
    # prepare the cross-validation procedure
    K = 10
    cv = StratifiedKFold(n_splits=K, random_state=1, shuffle=True)
    # create model
    model = LogisticRegression()
    # evaluate model
    scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
               'fp': make_scorer(fp), 'fn': make_scorer(fn),
               'auc': make_scorer(auc_),
               'acc': make_scorer(accuracy_score),
               'prec': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1': make_scorer(f1_score)}
    scores = cross_validate(model, X, y, scoring=scoring, cv=cv, n_jobs=-1)
    TN = mean(scores['test_tn'].ravel())
    FP = mean(scores['test_fp'].ravel())
    # report performance
    print('Accuracy: %.3f' % (mean(scores['test_acc'])))
    print('Precision: %.3f' % (mean(scores['test_prec'])))
    print('Recall: %.3f' % (mean(scores['test_recall'])))
    print('Recall-: %.3f' % (TN / (TN + FP)))
    print('AUC: %.3f' % (mean(scores['test_auc'])))
    print('F1: %.3f' % (mean(scores['test_f1'])))
    print("-----------------------------")
    print(f"Mean Confusion Matrix: TN: {mean(scores['test_tn'].ravel()):3f}, FP: {mean(scores['test_fp'].ravel()):3f}, FN: {mean(scores['test_fn'].ravel()):3f}, TP: {mean(scores['test_tp'].ravel()):3f}")
