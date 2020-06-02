import pandas as pd
import numpy as np
import glob
import os

%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve



class functionals:
    
    # instance attributes
    def __init__(self, path, indir, outdir):
        self.path = path
        self.fig_id = fig_id
        self.orig_headers = [
    'CREDIT SCORE', 'FIRST PAYMENT DATE', 'FIRST TIME HOMEBUYER FLAG', 'MATURITY DATE',
    'MSA', 'MI %', 'NUMBER OF UNITS', 'OCCUPANCY STATUS', 'ORIGINAL CLTV',
    'ORIGINAL DTI', 'ORIGINAL UPB', 'ORIGINAL LTV', 'ORIGINAL INTEREST RATE',
    'CHANNEL', 'PPM FLAG', 'PRODUCT TYPE', 'PROPERTY STATE', 'PROPERTY TYPE', 
    'POSTAL CODE', 'LOAN SEQUENCE NUMBER', 'LOAN PURPOSE', 'ORIGINAL LOAN TERM',
    'NUMBER OF BORROWERS', 'SELLER NAME', 'SERVICER NAME', 'UNKNOWN'
]
    self.mp_headers = [
    'LOAN SEQUENCE NUMBER', 'MONTHLY REPORTING PERIOD', 'CURRENT ACTUAL UPB',
    'CURRENT LOAN DELINQUENCY STATUS', 'LOAN AGE', 'REMAINING MONTHS TO LEGAL MATURITY',
    'REPURCHASE FLAG', 'MODIFICATION FLAG', 'ZERO BALANCE CODE',
    'ZERO BALANCE EFFECTIVE DATE', 'CURRENT INTEREST RATE', 'CURRENT DEFERRED UPB', 
    'DDLPI', 'MI RECOVERIES', 'NET SALES PROCEEDS', 'NON MI RECOVERIES', 'EXPENSES', 
    'LEGAL COSTS', 'MAINTENANCE AND PRESERVATION COSTS', 'TAXES AND INSURANCE', 
    'MISCELLANEOUS EXPENSES', 'ACTUAL LOSS CALCULATION', 'MODIFICATION COST',
    'STEP MODIFICATION FLAG', 'DEFERRED PAYMENT MODIFICATION', 'ELTV', 'ZERO BALANCE REMOVAL UPB',
    'DELINQUENT ACCRUED INTEREST'
]
    self.indir = indir
    self.outdir = outdir
        
        
    # save figure
    def save_fig(self, self.fig_id, tight_layout=True):
        self.path = os.path.join(self.fig_id + ".png") 
        print("Saving figure", self.fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(self.path, format='png', dpi=300)

    # concatenate origin data
    def orig_concatter(self):
        self.files = glob.glob(self.path)

        with open('orig.txt','w') as result:
            for file_ in files:
                for line in open(file_, 'r'):
                    result.write(line)

    # concatenate monthly performance data
    def mp_concatter(self, self.indir, self.outdir):
        self.filelist = glob.glob(self.indir)
        self.dflist = []
        for file in filelist:
            print(file)
            self.data = pd.read_csv(file, delimiter='|', names=self.mp_headers, low_memory=False)
            self.data = self.data[self.cols_p]
            self.data.dropna(inplace=True)
            self.dflist.append(self.data)
        self.concatdf = pd.concat(dflist, axis=0)
        self.concatdf.to_csv(self.outdir, index=False)
        return self.concatdf
        
        
    def data_cleaner(self, self.mp, self.orig):
        self.data = self.mp.set_index('LOAN SEQUENCE NUMBER').join(self.orig.set_index('LOAN SEQUENCE NUMBER'))
        self.data.dropna(inplace=True)
        self.data = self.data[(self.data['CREDIT SCORE'] >= 301) & (self.data['CREDIT SCORE'] <= 850)]
        self.data = self.data[(self.data['ORIGINAL CLTV'] >= 0) & (self.data['ORIGINAL CLTV'] <= 200)]
        self.data = self.data[(self.data['ORIGINAL DTI'] >= 0) & (self.data['ORIGINAL DTI'] <= 65)]
        self.data['MI %'].replace(999, 0, inplace=True)
        self.data['ZERO BALANCE CODE'].replace({1: 0, 9: 1, 6: 1, 3: 1, 2: 1, 15: 1}, inplace=True)
        return self.data
    
    
    def unzipper(self):
        pass