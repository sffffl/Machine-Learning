from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('nba.csv',delim_whitespace=True)
print(data.head())

minmax_scaler = MinMaxScaler()
# 标准化数据
#X = minmax_scaler.fit_transform(data.iloc[:,1:])
X = minmax_scaler.fit_transform(data.iloc[:,1:])
#print(X[:5])

# 肘部法则
loss = []
for i in range(2, 10):
    model = KMeans(n_clusters=i).fit(X)
    loss.append(model.inertia_)

plt.plot(range(2, 10), loss)
plt.xlabel('k')
plt.ylabel('loss')
plt.show()

k = 4
model = KMeans(n_clusters=k).fit(X)

# 将标签整合到原始数据上
data['clusters'] = model.labels_

for i in range(k):
    print('clusters:',i)
    label_data = data[data['clusters'] == i].iloc[:,0]
    print(label_data.values)