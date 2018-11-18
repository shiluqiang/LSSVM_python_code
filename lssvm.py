# -*- coding: utf-8 -*-
"""
Created on Wed May 16 12:48:56 2018

@author: lj
"""

from numpy import *

def loadDataSet(filename):
    '''导入数据
    input: filename:文件名
	output:dataMat(list)样本特征
	       labelMat(list)样本标签
    '''
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat
           

def kernelTrans(X,A,kTup):
    '''数据集中每一个数据向量与数据A的核函数值
    input: X--特征数据集
           A--输入向量
           kTup--核函数参量定义
    output: K--数据集中每一个数据向量与A的核函数值组成的矩阵
    '''
    X = mat(X)
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K/(-1 * kTup[1] ** 2))
    else: raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K
    
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.K = mat(zeros((self.m,self.m)))  #特征数据集合中向量两两核函数值组成的矩阵，[i,j]表示第i个向量与第j个向量的核函数值
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)
            

def leastSquares(dataMatIn,classLabels,C,kTup):
    '''最小二乘法求解alpha序列
    input:dataMatIn(list):特征数据集
          classLabels(list):分类标签集
          C(float):参数，（松弛变量，允许有些数据点可以处于分隔面的错误一侧)
          kTup(string): 核函数类型和参数选择 
    output:b(float):w.T*x+b=y中的b
           alphas(mat):alphas序列      
    '''
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,kTup)
	##1.参数设置
    unit = mat(ones((oS.m,1)))  #[1,1,...,1].T
    I = eye(oS.m)
    zero = mat(zeros((1,1)))
    upmat = hstack((zero,unit.T))
    downmat = hstack((unit,oS.K + I/float(C)))
	##2.方程求解
    completemat = vstack((upmat,downmat))  #lssvm中求解方程的左边矩阵
    rightmat = vstack((zero,oS.labelMat))    # lssvm中求解方程的右边矩阵
    b_alpha = completemat.I * rightmat
    oS.b = b_alpha[0,0]
    for i in range(oS.m):
        oS.alphas[i,0] = b_alpha[i+1,0]
    return oS.alphas,oS.b,oS.K

def predict(alphas,b,dataMat,testVec):
    '''预测结果
    input:alphas(mat):Lagrange乘子序列
          b(float):分隔超平面的偏置
          dataMat()
          
    output：sign(float(predict_value))(int):预测样本的类别
    '''
    Kx = kernelTrans(dataMat,testVec,kTup)   #可以对alphas进行稀疏处理找到更准确的值
    predict_value =  Kx.T * alphas + b
#    print('预测值为：%f'%predict_value)
#    print('分类结果为：%f'%sign(float(predict_value)))
    return sign(float(predict_value))

 


if __name__ == '__main__':
    ##1.导入数据
    print('-----------------------------1.Load Data-------------------------------')
    dataMat,labelMat = loadDataSet('testSetRBF.txt')
    C = 0.6
    k1 = 0.3
    kernel = 'rbf'
    kTup = (kernel,k1)
    ##2.训练模型
    print('----------------------------2.Train Model------------------------------')
    alphas,b,K = leastSquares(dataMat,labelMat,C,kTup)
    ##3.计算训练误差
    print('----------------------------3.Calculate Train Error--------------------')
    error = 0.0
    for i in range(len(dataMat)):
        test = predict(alphas,b,dataMat,dataMat[i])
        if test != float(labelMat[i]):
            error +=1.0
    errorRate = error/len(dataMat)
    print('---------------训练误差为：%f-------------------'%errorRate)




