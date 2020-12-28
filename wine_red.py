import pandas as pd
import numpy as np
data = pd.read_csv('./winequality-red.csv',sep=';',encoding='ISO-8859-1')

import sklearn 
from sklearn import linear_model
lm = linear_model.LogisticRegression()
features = data.columns.values.tolist()[:-1]
#x = data[features]

x = data.drop('quality',axis = 1)
y = data['quality']

print(data.head())

# fixed acidity;"volati
from sklearn.model_selection import cross_val_score
#logistic 中的scroing参数指定为accuracy
scores = cross_val_score(lm,x,y,cv=5,scoring='accuracy')
print(np.mean(scores))

import matplotlib.pyplot as plt
import seaborn as sns

#data = data.corr()
#sns.heatmap(data)
#plt.show()
data['quality'].value_counts()
from imblearn.over_sampling import RandomOverSampler
x = data.iloc[:,:-1].values #icloc方法根据位置悬着，即选择所有行，所有列去掉右数第一列
y = data['quality'].values
ros = RandomOverSampler()#构造采样方法
x,y = ros.fit_sample(x,y)
print(pd.DataFrame(y)[0].value_counts().sort_index())

from sklearn.model_selection import cross_val_score
#logistic 中的scroing参数指定为accuracy
scores = cross_val_score(lm,x,y,cv=5,scoring='accuracy')
print(np.mean(scores))

from sklearn import ensemble
#设置随机深林分类模型
rf = ensemble.RandomForestClassifier(100) #设置100个决策树
from sklearn.model_selection import cross_val_score
score = cross_val_score(rf,x,y,cv=5,scoring='accuracy')
print(np.mean(score))

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=111, stratify=y)
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100)) #声明超参数
hyperparameters = {'randomforestregressor__max_features':['auto','sqrt','log2'], 'randomforestregressor__max_depth':[None,5,3,1]} #优化模型
clf = GridSearchCV(pipeline,hyperparameters,cv=10)
clf.fit(x_train,y_train) #评估模型及预测
pred = clf.predict(x_test)
print("测试集ACC：")
print(r2_score(y_test, pred))
print("测试集MSE：")
print(mean_squared_error(y_test, pred))
