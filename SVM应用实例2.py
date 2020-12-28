import numpy as np #矩阵运算
import matplotlib.pyplot as pl #画图
from sklearn import svm

#固定每次抓取的值
np.random.seed(0)
#随机产生均值为20，方差为2的正太分布数据
X = np.r_[np.random.randn(20,2)-(2,2),np.random.randn(20,2)+(2,2)]
#对训练数据进行分类
Y = [0]*20 + [1]*20
print(X)
print(Y)


clf = svm.SVC(kernel = 'linear')
clf.fit(X,Y)

w = clf.coef_[0]
#计算斜率
a = -w[0]/w[1]
#产生连续的X值
x = np.linspace(-5,5)
#点斜式方程
y = a*x - (clf.intercept_[0]/w[1])

c = clf.support_vectors_[0]
y_down = a*x +(c[1]- a*c[0])
c1 = clf.support_vectors_[-1]
y_up = a*x + (c1[1] -a*c1[0])


print('w:',w)
print('a:',a)
print('support_vector:',clf.support_vectors_)

pl.plot(x,y,'k-')
pl.plot(x,y_down,'k--')
pl.plot(x,y_up,'k--')

pl.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1])
pl.scatter(X[:,0],X[:,1],c=Y)
pl.show()
