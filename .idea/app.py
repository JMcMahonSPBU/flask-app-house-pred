#!/usr/bin/env python
# coding: utf-8

# ## Predicting Real Estate Data famous Kaggle competition
# Creating a model based on the famous (2nd most popular in history, I think) Sberbank Kaggle competition.
#
# ### Main objectives
# -	 machine learning to solve price prediction problem
# -   Calculate metrics to know when model is ready for prod
#
# ### Tasks
# -	Encode dataset
# -	Split dataset to train and validation datasets
# -	Apply decision tree algorithm to build ML (machine learning) model for price predictions
# -   Calculate metrics
# -   Try other algorithms and factors to get a better solution
#

# ### 1. Load data with real estate prices

# In[374]:


# let's import pandas library and set options to be able to view data right in the browser
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.style as style
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
style.use('fivethirtyeight')
import numpy as np

# In[375]:


train_df = pd.read_csv('train.csv')

# In[376]:


# train_df.head()


# In[377]:


dtypes_dict = {}
col_names = train_df.columns
dtype_list = list(train_df.dtypes)

# In[378]:


# col_names[0]


# In[379]:


# dtype_list[0]


# In[380]:


for i in range(len(train_df.dtypes)):
    dtypes_dict[col_names[i]] = dtype_list[i]

# In[381]:


dtypes_dict

# In[382]:


train_df_sel = train_df[['id', 'num_room', 'metro_min_walk', 'kremlin_km', 'price_doc', 'full_sq', 'life_sq', 'floor',
                         'university_top_20_raion', 'railroad_station_walk_min', 'big_church_km', 'cafe_avg_price_500',
                         'cafe_count_500_price_high']]

# In[383]:


# train_df_sel.head()


# In[384]:


# train_df_sel.full_sq.plot(title='full_sq')


# In[385]:


# train_df_sel.life_sq.plot(title='life_sq')


# In[386]:


# train_df_sel.railroad_station_walk_min.plot(title='railroad_station_walk_min')


# In[387]:


# train_df_sel.cafe_count_500_price_high.plot(title='cafe_count_500_price_high')


# In[388]:


# train_df_sel.price_doc.plot(title='price_doc')


# In[389]:


# train_df_sel.info()


# In[390]:


# train_df_sel.num_room.fillna(train_df_sel.num_room.mode(), inplace=True)
train_df_sel.metro_min_walk.fillna(train_df_sel.metro_min_walk.median())
train_df_sel.kremlin_km.fillna(train_df_sel.kremlin_km.mean())
train_df_sel.full_sq.fillna(train_df_sel.full_sq.median())
train_df_sel.life_sq.fillna(train_df_sel.life_sq.median(), inplace=True)
train_df_sel.floor.fillna(train_df_sel.floor.mode())
train_df_sel.university_top_20_raion.fillna(train_df_sel.university_top_20_raion.mode())
train_df_sel.railroad_station_walk_min.fillna(train_df_sel.railroad_station_walk_min.median())
train_df_sel.big_church_km.fillna(train_df_sel.big_church_km.median())
# train_df_sel.cafe_avg_price_500.fillna(train_df_sel.cafe_avg_price_500.median())
train_df_sel.cafe_count_500_price_high.fillna(train_df_sel.cafe_count_500_price_high.median())

# In[391]:


# train_df_sel.info()


# In[392]:


train_df_sel = train_df_sel.drop(columns=['num_room', 'cafe_avg_price_500'])

# In[393]:


# train_df_sel.info()


# In[394]:


train_df_sel = train_df_sel.dropna()

# In[395]:


# train_df_sel.info()


# ### Create datasets training, testing and a holdout dataset.
# Kaggle data already has train and test and our holdout is the competition held out data

# In[396]:


# going to convert price to log because competitoin participants had some success with this

train_df_sel['price_doc'] = np.log(train_df_sel['price_doc'])

# In[397]:


# train_df_sel['price_doc'].hist()


# In[398]:


## Building Decision Tree model


# In[399]:


from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

# In[400]:


# type(X_train)


# In[401]:


# test_df = pd.read_csv('test.csv')


# In[402]:


# test_df.head()


# In[403]:


# for i in test_df.columns:
#    print(i)


# In[404]:


# test_df_sel = test_df[['id', 'num_room','metro_min_walk','kremlin_km', 'price_doc', 'full_sq', 'life_sq', 'floor', 'university_top_20_raion', 'railroad_station_walk_min', 'big_church_km', 'cafe_avg_price_500', 'cafe_count_500_price_high']]


# # someone deleted the target from the test dataset, so turning the train set into holdout and test sets....
#
# went back and added timestamp so I could split

# In[405]:


# train_df_sel['timestamp']


# In[406]:


from sklearn.model_selection import train_test_split

# Let's say we want to split the data in 80:10:10 for train:valid:test dataset
train_size = 0.8

X = train_df_sel.drop(columns=['price_doc']).copy()
y = train_df_sel['price_doc']

# In[407]:


# In the first step we will split the data in training and remaining dataset
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8)

# Now since we want the valid and test size to be equal (10% each of overall data).
# we have to define valid_size=0.5 (that is 50% of remaining data)
test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

# print(X_train.shape), print(y_train.shape)
# print(X_valid.shape), print(y_valid.shape)
# print(X_test.shape), print(y_test.shape)


# In[408]:


y_train = y_train.values.reshape(-1, 1)
y_valid = y_valid.values.reshape(-1, 1)
# print(X_train)


# In[409]:


from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_valid = sc_X.fit_transform(X_valid)
y_train = sc_y.fit_transform(y_train)
y_valid = sc_y.fit_transform(y_valid)

# In[410]:


# X_train.shape, y_train.shape


# In[411]:


tree = DecisionTreeRegressor(max_depth=3, random_state=17)
tree.fit(X_train, y_train)

# In[412]:


tree_predictions = tree.predict(X_valid)

# In[413]:


# tree_predictions[:5]


# In[414]:


predictions = sc_y.inverse_transform(tree_predictions)
values = y_valid
predictions = predictions.reshape(-1, 1)
# for pred, val in zip(predictions,values):
# print("Prediction: {}, True Value {}".format(pred, val))


# In[415]:


# print('MAE:', metrics.mean_absolute_error(y_valid, tree_predictions))
# print('MSE:', metrics.mean_squared_error(y_valid, tree_predictions))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, tree_predictions)))


# In[416]:


##Building linear regression


# In[417]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
# print('MAE:', metrics.mean_absolute_error(y_valid, model.predict(X_valid)))
# print('MSE:', metrics.mean_squared_error(y_valid, model.predict(X_valid)))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, model.predict(X_valid))))


# In[418]:


##Building random forest regressor


# In[420]:


random_forest_model = RandomForestRegressor(n_estimators=15,
                                            bootstrap=0.8,
                                            max_depth=15,
                                            min_samples_split=3,
                                            max_features=1)
random_forest_model.fit(X_train, y_train)
predictions = random_forest_model.predict(X_valid)
# print('MAE:', metrics.mean_absolute_error(y_valid, predictions))
# print('MSE:', metrics.mean_squared_error(y_valid, predictions))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, predictions)))


# In[421]:
predictions = predictions.reshape(-1, 1)

# plt.figure(figsize=(16,8))
# plt.plot(sc_y.inverse_transform(y_valid),label ='Test', color= 'blue')
# plt.plot(sc_y.inverse_transform(predictions), label = 'predict', color = 'orange')
# plt.show()


# In[422]:


##Save the model, encoder and the scaler as a pipeline


# In[423]:


import joblib

# In[424]:


model_file = 'model_sber.pkl'
scaler_x = 'scaler_x_sber.pkl'
scaler_y = 'scaler_y_sber.pkl'
joblib.dump(sc_X, scaler_x)
joblib.dump(sc_y, scaler_y)
joblib.dump(random_forest_model, model_file)