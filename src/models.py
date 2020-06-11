import pandas as pd
import numpy as np
import glob
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
mpl.rc('axes', labelsize=18)
mpl.rc('xtick', labelsize=15)
mpl.rc('ytick', labelsize=15)
mpl.rc('legend',**{'fontsize':16.5})
mpl.rc('lines', linewidth=2)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import xgboost as xgb
from xgboost import XGBRegressor


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(fig_id + ".png") 
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)   


def model_train_test(X_train, X_test, y_train, y_test, X_, y_, title_train, title_test, save_train, save_test):
    
    # logreg
    pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(random_state=99, max_iter=1000))
    pipe_lr.fit(X_train, y_train)
    test_probas_lr = pipe_lr.predict_proba(X_test)[:,1]
    
    # rand forest
    pipe_rfc = make_pipeline(StandardScaler(), RandomForestClassifier(
        oob_score=True, n_jobs=-1, max_depth=5, random_state=99))
    pipe_rfc.fit(X_train, y_train)
    test_probas_rfc = pipe_rfc.predict_proba(X_test)[:,1]
    
    # gradboost
    pipe_gbc = make_pipeline(StandardScaler(), GradientBoostingClassifier(learning_rate=0.2, random_state=99))
    pipe_gbc.fit(X_train, y_train)
    test_probas_gbc = pipe_gbc.predict_proba(X_test)[:,1]
    
    # naive
    pipe_nb = make_pipeline(StandardScaler(), BernoulliNB())
    pipe_nb.fit(X_train, y_train)
    test_probas_nb = pipe_nb.predict_proba(X_test)[:,1]
    
    # XGBoost
    pipe_xgb = make_pipeline(StandardScaler(), XGBRegressor(objective='binary:logistic', eval_metric='auc'))
    pipe_xgb.fit(X_train, y_train)
    test_xgb = pipe_xgb.predict(X_test)
    
    # graph
    
    # generate a no skill prediction
    ns_probs = [0 for _ in range(len(y_test))]

    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, test_probas_lr)
    rfc_auc = roc_auc_score(y_test, test_probas_rfc)
    gbc_auc = roc_auc_score(y_test, test_probas_gbc)
    nb_auc = roc_auc_score(y_test, test_probas_nb)
    xgb_auc = roc_auc_score(y_test, test_xgb)

    ns = 'No Skill ROC-AUC score: %.3f' % ns_auc
    lr = 'Logistic Regression ROC-AUC score: %.3f' % lr_auc
    rfc = 'Random Forest Classifier ROC-AUC score: %.3f' % rfc_auc
    gbc = 'Gradient Boost ROC-AUC score: %.3f' % gbc_auc
    nb = 'Bernoulli NB ROC-AUC score: %.3f' % nb_auc
    xgb = 'XGBoost ROC-AUC score: %.3f' % xgb_auc

    print(ns)
    print(lr)
    print(nb)
    print(rfc)
    print(gbc)
    print(xgb)

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, test_probas_lr)
    rfc_fpr, rfc_tpr, _ = roc_curve(y_test, test_probas_rfc)
    gbc_fpr, gbc_tpr, _ = roc_curve(y_test, test_probas_gbc)
    nb_fpr, nb_tpr, _ = roc_curve(y_test, test_probas_nb)
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, test_xgb)

    # fig size
    plt.figure(figsize=(16,8))

    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label=ns, color='blue')
    plt.plot(lr_fpr, lr_tpr, linestyle='--', label=lr, color='red')
    plt.plot(nb_fpr, nb_tpr, linestyle='--', label=nb, color='purple')
    plt.plot(rfc_fpr, rfc_tpr, linestyle='--', label=rfc, color='green')
    plt.plot(gbc_fpr, gbc_tpr, linestyle='--', label=gbc, color='orange')
    plt.plot(xgb_fpr, xgb_tpr, linestyle='--', label=xgb, color='black')
    
    plt.title(title_train, fontsize=18)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend(loc='lower right')
    # show the plot
    save_fig(save_train)
    plt.show()    

    test_probas_lrt = pipe_lr.predict_proba(X_)[:,1]
    test_probas_rfct = pipe_rfc.predict_proba(X_)[:,1]
    test_probas_gbct = pipe_gbc.predict_proba(X_)[:,1]
    test_probas_nbt = pipe_nb.predict_proba(X_)[:,1]
    test_xgbt = pipe_xgb.predict(X_)

    # generate a no skill prediction
    nst_probs = [0 for _ in range(len(y_))]

    # calculate scores
    nst_auc = roc_auc_score(y_, nst_probs)
    lrt_auc = roc_auc_score(y_, test_probas_lrt)
    rfct_auc = roc_auc_score(y_, test_probas_rfct)
    gbct_auc = roc_auc_score(y_, test_probas_gbct)
    nbt_auc = roc_auc_score(y_, test_probas_nbt)
    xgbt_auc = roc_auc_score(y_, test_xgbt)
    

    nst = 'No Skill ROC-AUC score: %.3f' % nst_auc
    lrt = 'Logistic Regression ROC-AUC score: %.3f' % lrt_auc
    rfct = 'Random Forest Classifier ROC-AUC score: %.3f' % rfct_auc
    gbct = 'Gradient Boost ROC-AUC score: %.3f' % gbct_auc
    nbt = 'Bernoulli NB ROC-AUC score: %.3f' % nbt_auc
    xgbt = 'XGBoost ROC-AUC score: %.3f' % xgbt_auc

    print(nst)
    print(lrt)
    print(nbt)
    print(rfct)
    print(gbct)
    print(xgbt)

    # calculate roc curves
    nst_fpr, nst_tpr, _ = roc_curve(y_, nst_probs)
    lrt_fpr, lrt_tpr, _ = roc_curve(y_, test_probas_lrt)
    rfct_fpr, rfct_tpr, _ = roc_curve(y_, test_probas_rfct)
    gbct_fpr, gbct_tpr, _ = roc_curve(y_, test_probas_gbct)
    nbt_fpr, nbt_tpr, _ = roc_curve(y_, test_probas_nbt)
    xgbt_fpr, xgbt_tpr, _ = roc_curve(y_, test_xgbt)

    # fig size
    plt.figure(figsize=(16,8))

    # plot the roc curve for the model
    plt.plot(nst_fpr, nst_tpr, linestyle='--', label=nst, color='blue')
    plt.plot(lrt_fpr, lrt_tpr, linestyle='--', label=lrt, color='red')
    plt.plot(nbt_fpr, nbt_tpr, linestyle='--', label=nbt, color='purple')
    plt.plot(rfct_fpr, rfct_tpr, linestyle='--', label=rfct, color='green')
    plt.plot(gbct_fpr, gbct_tpr, linestyle='--', label=gbct, color='orange')
    plt.plot(xgbt_fpr, xgbt_tpr, linestyle='--', label=xgbt, color='black')

    plt.title(title_test, fontsize=18)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend(loc='lower right')
    # show the plot
    save_fig(save_test)
    plt.show()
    
    