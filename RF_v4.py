
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


# import geopandas as geopandas

# --- <for csv inputs> ---
fields=['LandUse','LandForm','SoilTypeAndThickness','Geology','Slope','Aspect','SPI','TWI','STI','Rainfall','DistanceToWaterways','Class']


dataset = pd.read_csv(r'F:\1.EDU\1.UCSC\4th year\RESEARCH\research!\nbro\Data Set\1.data for model\total data\FeaturesForAllPoints.csv',usecols=fields)

#dataset = pd.read_csv(r'C:\Users\Dell\Desktop\dataFile.csv',usecols=fields)


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
X = dataset.iloc[:, 1:11].values

y=dataset.iloc[:,11].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1,shuffle='true')

# <find best parameters>

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
print(RandomForestClassifier())
print(RandomForestRegressor())

rfc = RandomForestClassifier()
# (n_estimators=100,criterion='entropy',max_depth=20,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',class_weight='balanced',bootstrap='true',random_state=0,oob_score='true')
parameters = {
    "n_estimators":[5,10,50,100,250],
    "max_depth":[2,4,8,16,32,None],
    "min_samples_split":[2],
    "min_samples_leaf":[2,4,8,16,32,None],
    "random_state":[0,1,2,4,8,16,32,None],
    "criterion":['entropy','gini'],
    "max_features":['sqrt','auto','log2',None],
    "class_weight":['balanced'],
    "bootstrap":['true',]
    
}

from sklearn.model_selection import GridSearchCV
cv = GridSearchCV(rfc,parameters,cv=5)
cv.fit(X_train,y_train.ravel())

def display(results):
    # print str(results)

    print ('Best parameters are:'+ str(results.best_params_))
    print("\n")
    # mean_score = results.cv_results_['mean_test_score']
    # std_score = results.cv_results_['std_test_score']
    # params = results.cv_results_['params']
    # for mean,std,params in zip(mean_score,std_score,params):
    #     print(f"{round(mean,3)} + or -{round(std,3)} for the {params}")

display(cv)

# </find best parameters>

# classifier=RandomForestClassifier(n_estimators=40,criterion='entropy',max_depth=20,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',class_weight='balanced',bootstrap='true',random_state=0,oob_score='true')
# classifier.fit(X_train,y_train)
# y_pred=classifier.predict(X_test)

# cm=confusion_matrix(y_test,y_pred)
# print(cm)


# def plot_roc_cur(fper, tper):  
#     plt.plot(fper, tper, color='orange', label='ROC')
#     plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend()
#     plt.show()

# probs = classifier.predict_proba(X_test)
# probs = probs[:, 1]
# fper, tper, thresholds = roc_curve(y_test, probs) 
# plot_roc_cur(fper, tper)

# print("Accuracy on test data: {:.4f}".format(classifier.score(X_test, y_test)))

