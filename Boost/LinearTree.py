#!/usr/bin/python 
# -*- coding: utf-8 -*-
#
# Decision Tree
# HW-1, sphere.mail.ru
#
# Author: Rakitin Vitaliy
# vitaliyrakitin@ya.ru
#

import numpy as np
import sys
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR

class Node(object):
    ''' A node of a Tree '''

    def __init__(self, mapping, max_depth):

        self.mapping = mapping
        self.max_depth = max_depth

        self.left = None
        self.right = None

        self.finished_value = None
        self.params = None
        self.value = None
        
        
    def _decision_indexes(self, data):
        '''
        Count the decision mask for the data
        (left node - True, right node - False)

        Parameters:
            * data (np.array)

        Returns:
            * np.array mask of True/False values 
        '''
        
        #data1  = np.concatenate([data,  np.ones((len(data),1))], axis=1)
        #return np.sum(data1 * self.params, axis = 1) < self.value
        return  self.reg.predict(data) < self.value

                
    def _count_optimal_splitting(self, data, target):
        '''
        Count optimal splitting of the data
        
        '''
        #params, predicted_data = self.regression(data, target) 
        self.reg = LinearRegression().fit(data, target)
        predicted_data = self.reg.predict(data)
        params = None
        value = self._minimizing_mse(target,predicted_data)
        return params, value
    
       
    def _minimizing_mse(self, target, predicted_data = None):
        '''
        Minimizing MSE on predicred results of Linear Regression
        '''
        args = np.argsort(predicted_data)
        loss = sys.maxsize
        best_ind = None
        prev_el = None
        length = target.shape[0]
        
        for ind, el in enumerate(predicted_data[args]):
            if el == prev_el:
                continue
                
            prev_el = el
            
            cur_loss = self.MSE(target[args][:ind]) * ind + (length - ind) * self.MSE(target[args][ind:])

            if cur_loss < loss:
                loss = cur_loss
                best_ind = ind

        return predicted_data[args][best_ind]
    

    @staticmethod
    def MSE(target, predicted = None, is_mean = False):
        ''' 
        MSE criterion 
        
        Parameters:
            * target (np.array)
            * predicted (np.array) - predicted target (Mean if None)
              Default: None
            * is_mean - divide on the target len or not
              Default: False

        '''
        if predicted is None:
            if len(target) > 0:
                predicted = np.mean(target)
        if is_mean:
            MSE = ((predicted - target)**2).sum()  / float(len(target))
        else:
            MSE = ((predicted - target)**2).sum()
        return MSE
    
    
    @staticmethod
    def regression(data, target):
        data1 = np.concatenate([data,  np.ones((len(data),1))], axis=1)
        w = np.linalg.inv(data1.T.dot(data1)).dot(data1.T).dot(target)
        return w, data1.dot(w)
    
    
    def fit(self, data, target):
        '''
        Fitting model

        Parameters:
            * data (np.array)
            * target (np.array)
            * mask (np.array with data shape) - argsort for every feature of data 

        Returns:
            * self
        '''
                
        if np.unique(target[self.mapping]).shape[0] < 2 or self.max_depth < 1:
            self.finished_value = np.mean(target[self.mapping])
            return self
        
        self.params, self.value = self._count_optimal_splitting(data[self.mapping], target[self.mapping])
        mapping = self._decision_indexes(data)
        
        if mapping[self.mapping][mapping[self.mapping] == True].shape[0] < 1 \
           or mapping[self.mapping][mapping[self.mapping] == False].shape[0] < 1:
            
            print ("OOOOOOOPS: self.vaue is None!")
            self.finished_value = np.mean(target[self.mapping])
        
        else:
            self.left = Node(mapping * self.mapping, self.max_depth - 1).fit(data, target)
            self.right = Node((mapping == False) * self.mapping, self.max_depth - 1).fit(data, target)
        
        return self
    
    
    def predict(self, data):
        ''' Prediction '''
        
        target = np.zeros(data.shape[0])

        # if this is the last node of the branch
        if self.finished_value is not None:
            target += self.finished_value
        
        else:

            mapping = self._decision_indexes(data)
            if data[mapping].shape[0] > 0:
                target[mapping] = self.left.predict(data[mapping])
            if data[mapping == False].shape[0] > 0:
                target[mapping == False] = self.right.predict(data[mapping == False])

        return target


class LinearTree(object):
    ''' Decision Tree '''
    def __init__(self, max_depth = 4):
        '''
        Parameters:
            * data (np.array)
            * target (np.array)
            * max_depth (int)
              Default: 4
        '''
        
        self.max_depth = max_depth    
    

    def fit(self, data, target):
        ''' Fitting model '''

        self.Node = Node(np.ones(len(target)) == 1, self.max_depth)
        self.Node.fit(data, target)
        return self


    def predict(self, data):
        ''' Prediction '''
        return self.Node.predict(data)