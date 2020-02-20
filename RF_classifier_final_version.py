# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 13:13:20 2020

@author: Dell
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


# import geopandas as geopandas

# --- <for csv inputs> ---
# fields=['LandUse','LandForm','SoilTypeAndThickness','Geology','Slope','Aspect','SPI','TWI','STI','Rainfall','DistanceToWaterways','Class']
feilds_x=['LandForm','SoilTypeAndThickness','Geology','Slope','Aspect','SPI','TWI','STI','Rainfall','DistanceToWaterways']
# feilds_x=['LandUse','LandForm','SoilTypeAndThickness','Geology','Slope','Aspect','SPI','TWI','STI','Rainfall']
feilds_y=['Class']

dataset = pd.read_csv(r'F:\1.EDU\1.UCSC\4th year\RESEARCH\research!\nbro\Data Set\1.data for model\total data\FeaturesForAllPoints.csv')


def convert(data):
    #data=data.fillna(-999)
    encode_data = preprocessing.LabelEncoder()    
    dataset['LandUse'] = encode_data.fit_transform(data.LandUse.astype(str))
    dataset['LandForm'] = encode_data.fit_transform(data.LandForm.astype(str))
    dataset['SoilTypeAndThickness'] = encode_data.fit_transform(data.SoilTypeAndThickness.astype(str))
    dataset['Geology'] = encode_data.fit_transform(data.Geology.astype(str))
    dataset['Slope'] = encode_data.fit_transform(data.Slope.astype(str))
    dataset['Aspect'] = encode_data.fit_transform(data.Aspect.astype(str))
    dataset['SPI'] = encode_data.fit_transform(data.SPI.astype(str))
    dataset['TWI'] = encode_data.fit_transform(data.TWI.astype(str))
    dataset['STI'] = encode_data.fit_transform(data.STI.astype(str))
    dataset['Rainfall'] = encode_data.fit_transform(data.Rainfall.astype(str))
    dataset['DistanceToWaterways'] = encode_data.fit_transform(data.DistanceToWaterways.astype(str))    
    data=data.fillna(-999)
    return data

dataset=convert(dataset)
X = dataset.loc[:, feilds_x]
y=dataset.loc[:,feilds_y]
# print(X.shape,y.shape)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1,shuffle='true')

# classifier=RandomForestClassifier(n_estimators=25,criterion='entropy',max_depth=20,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',class_weight='balanced',bootstrap='true',random_state=0,oob_score='true')
classifier=RandomForestClassifier(n_estimators=35,criterion='entropy',max_depth=30,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',class_weight='balanced',bootstrap='true',random_state=0,oob_score='true',max_leaf_nodes=7)
# classifier=RandomForestClassifier(n_estimators=40,criterion='entropy',max_depth=20,min_samples_split=7,min_samples_leaf=1,max_features='sqrt',max_leaf_nodes=15,class_weight='balanced',bootstrap='true',random_state=0,oob_score='true')

classifier.fit(X_train,y_train)
y_predTr=classifier.predict(X_train)
y_pred=classifier.predict(X_test)

cm1=confusion_matrix(y_train,y_predTr)
print(cm1)
cm2=confusion_matrix(y_test,y_pred)
print(cm2)


def plot_roc_cur(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

probs = classifier.predict_proba(X_test)
probs = probs[:, 1]
fper, tper, thresholds = roc_curve(y_test, probs) 
plot_roc_cur(fper, tper)

from sklearn import  metrics
auc= metrics.roc_auc_score(y_test, probs)
print("area under cureve: {:.4f}". format(metrics.roc_auc_score(y_test, probs)))

print("Accuracy on Traning data: {:.4f}".format(classifier.score(X_train,y_train)))
print("Accuracy on test data: {:.4f}".format(classifier.score(X_test, y_test)))

