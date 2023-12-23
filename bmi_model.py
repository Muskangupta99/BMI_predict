import numpy as np
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
data = pd.read_csv(r'D:\Datasets\500_Person_Gender_Height_Weight_Index.csv')
def convert_gender_to_label(x):
    if x['Gender'] == 'Male':
        return 1
    elif x['Gender'] == 'Female':
        return 0
def convert_height(x):
    a=x['Height']
    a /= 100.
    return a
data['Gender'] = data.apply(convert_gender_to_label,axis=1)
data['Height'] = data.apply(convert_height,axis=1)
X_scaled= data.iloc[:,:3]
y=data['Index']
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y, test_size = 0.2, random_state = 0)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
def run_randomForest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=300,criterion='entropy', random_state=0,n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
model = RFE(GradientBoostingClassifier(n_estimators=300, random_state=0), n_features_to_select = 3)
model.fit(X_train, y_train)
X_train_rfe = model.transform(X_train)
X_test_rfe = model.transform(X_test)
run_randomForest(X_train_rfe, X_test_rfe, y_train, y_test)
joblib.dump(model, "model.pkl")