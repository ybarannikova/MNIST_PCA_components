#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 21:27:35 2017

@author: ybarannikova
"""

import numpy as np
import numpy.linalg as LN
import pandas as pd
import matplotlib.pyplot as plt


def commonDirection(allData):
    covMat=np.matmul(allData.T,allData)
    w,v = LN.eig(covMat)
    component=v[:,0].real
    return component
    
def dataProjection(dataVector,component):
    return np.dot(dataVector,component)
    
def componentRemoval(xOld,component,z):
    xNew = xOld-component*z
    return xNew
    
def findComponents(allData, n):
    allData = allData.apply(lambda c:c-np.mean(c), axis=0)
    topComponents = []
    for i in range(n):
        component = commonDirection(allData)
        topComponents.append(component)
        allData = allData.apply(lambda v: componentRemoval(
                v,component,dataProjection(v,component)),axis=1)
    return topComponents
    

digits = pd.read_csv("mnist_train.csv", header=None)
allData = digits.iloc[:,1:]
topComponents=findComponents(allData,3)
for i in topComponents:
    pic=i.reshape((28,28))
    plt.imshow(pic)
    plt.show()
    
    
    
'''
5.
First Principal Component represents two shapes: looks like a zero with a one 
inside. The blue shapes are consistent with number zero and the yellow with 
number one. So for example number eight will be in between but close to zero.
It is hard to describe what the second component looks like. You can see some
dominant features of number seven. Third component looks like a three. 
'''
