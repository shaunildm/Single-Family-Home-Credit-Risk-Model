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
    def __init__(self, path):
        self.path = path
        self.fig_id = fig_id
        
        
    # save figure
    def save_fig(self, fig_id, tight_layout=True):
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
    def mp_concatter(self, indir, outdir):
        filelist = glob.glob(indir)
        dflist = []
        for file in filelist:
            print(file)
            data = pd.read_csv(file, delimiter='|', names=mp_headers, low_memory=False)
            data = data[cols_p]
            data.dropna(inplace=True)
            dflist.append(data)
        concatdf = pd.concat(dflist, axis=0)
        concatdf.to_csv(outdir, index=False)