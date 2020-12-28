# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 18:05:20 2018

@author: Administrator
"""

from sklearn import svm

x = [[3,3],[4,3],[1,1]]
y = [1,1,-1]

clf = svm.SVC(kernel = 'linear')
clf.fit(x,y)

print(clf)
print(clf.support_vectors_)
print(clf.support_)
print(clf.n_support_)

predictlabel = clf.predict([[1,0]])
predictlabel_ = clf.predict([[3,8]])
print(predictlabel)
print(predictlabel_)

