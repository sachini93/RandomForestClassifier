import pandas as pd
import numpy as np
# import geopandas as geopandas

# --- <for csv inputs> ---
feilds=['Class','Land_Use','Land_Form','Soil_Ty_Thick','Geology','Slope','Aspect','SPI','TWI','STI','Rainfall']

dataset = pd.read_csv(r'F:\1.EDU\1.UCSC\4th year\RESEARCH\research!\nbro\Data Set\final outputs\final outputs\CSV files\FeaturesForAllPoints.csv',usecols=feilds)

from sklearn import preprocessing
def convert(data):
    encode_data = preprocessing.LabelEncoder()
    dataset['Land_Use'] = encode_data.fit_transform(data.Land_Use)
    dataset['Land_Form'] = encode_data.fit_transform(data.Land_Form)
    dataset['Soil_Ty_Thick'] = encode_data.fit_transform(data.Soil_Ty_Thick)
    dataset['Geology'] = encode_data.fit_transform(data.Geology)
    data=encode_data.fillna(-999)
    return data

data=convert(dataset)

print data.head()

X = dataset.iloc[:, 3:12].values
y = dataset.iloc[:, 2].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# #Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import accuracy_score




accuracy = regressor.score(y_test, y_pred)
print('Accuracy score:', accuracy )

# --- </for csv inputs> ---


# --- </for .shp inputs> ---

# file_path = r'F:\1.EDU\1.UCSC\4th year\RESEARCH\research!\nbro\Data Set\1.data for model'

# ds_litho = geopandas.read_file(file_path + '\litho\Lithointersected.shp')
# ds_lf = geopandas.read_file(file_path + '\LF\LFintersected1.shp')
# # ds_lu = geopandas.read_file(file_path + "LU\--")
# # ds_obThickness = geopandas.read_file(file_path + "obThickness\--")
# # ds_rainfall = geopandas.read_file(file_path + "rainfall\--")
# ds_rasters = geopandas.read_file(file_path + '\LS_rasters\LS_poly_kalutara.shp')
# ds_litho.head()

# --- </for .shp inputs> ---

