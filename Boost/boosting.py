#!/usr/bin/python 
# -*- coding: utf-8 -*-
#
# Gradient Boosting of Decision Trees
# HW-1, sphere.mail.ru
#
# Author: Rakitin Vitaliy
# vitaliyrakitin@ya.ru
#

import numpy as np
from .LinearTree import LinearTree
from tqdm import tqdm
from sklearn.tree import DecisionTreeRegressor

class GradientBoosting(object):
    ''' 
    Gradient Boosting with L = (y - h(x))**2 / 2
    dL/dh =  y - h(x)
    '''
    def __init__(self, n_estimators=10, max_depth=10, model = None):
        '''
        Parameters:
            * n_estimators (int) - estimators number
            * max_depth (int) of a tree
        '''
        
        self.estimators_list = []
        self.n_estimators = n_estimators
        self.max_depth = max_depth

        if model is not None:
            self.model = model
        else:
            self.model = LinearTree
        
        self.estimators_list = []
        self.b = []
        self.loss = []
        self.test_loss = []

    @staticmethod
    def MSE(target, predicted = None, is_test = False):
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
            predicted = np.mean(target)
        
        if is_test:
            return ((predicted - target)**2).sum() / predicted.shape[0]
        MSE = ((predicted - target)**2).sum() / 2
        return MSE

    
    def _count_b(self, target, current_predict, new_predict, step = 1e-1, max_b = 5000):
        '''
        Count optimal parameter b
        
        Parameters:
            * target 
            * prediction 
            * new_predicton
            * step (int) - length of the step to count b
            * max_b - max value of b
        
        Returns:
            * b
        '''
        b = step * 2
        loss = self.MSE(target, current_predict + step * new_predict)
        anti_loss = self.MSE(target, current_predict - step * new_predict)
        
        # determine should the parameter be negative or positive
        if anti_loss < loss:
            new_loss = self.MSE(target, current_predict - (b + step) * new_predict)
            
            while new_loss < loss and b < max_b:
                b += step
                loss = new_loss
                new_loss = self.MSE(target, current_predict - (b + step) * new_predict)
        
        else:
            new_loss = self.MSE(target, current_predict + (b + step) * new_predict)
            
            while new_loss < loss and b < max_b:
                b += step
                loss = new_loss
                new_loss = self.MSE(target, current_predict + (b + step) * new_predict)
                
        return b
        
    def fit(self, data, target, test_data=None, test_target=None, shrinkage = 0.1):
        ''' Fitting model '''
        self.estimators_list = []
        self.b = []
        self.loss = []
        self.test_loss = []

        # step 1. Initialization

        first_estimator = self.model(max_depth = self.max_depth).fit(data, target)     
        #first_estimator = DecisionTreeRegressor(max_depth = self.max_depth).fit(data, target)
        
        self.estimators_list.append(first_estimator)
        self.b.append(1)
        
        prediction = first_estimator.predict(data)
        self.loss.append(self.MSE(target, prediction, is_test = True))

        if test_data is not None:
            self.test_loss.append(self.MSE(test_target, self.predict(test_data), is_test = True))
        
        for i in tqdm(range(1, self.n_estimators)):
            
            # step 2.a Count antigrad
            antigrad = target - prediction
            
            # step 2.b count new base model
            new_estimator = self.model(max_depth=self.max_depth).fit(data, antigrad)

            #new_estimator = DecisionTreeRegressor(max_depth = self.max_depth).fit(data, antigrad)
             
            # step 2.c count parameter b for the model
            new_prediction = new_estimator.predict(data) 
            b = self._count_b(target, prediction, new_prediction)
            
            # step 2.d save estimator
            self.estimators_list.append(new_estimator)
            self.b.append(b * shrinkage)

            prediction += shrinkage * b * new_prediction
            self.loss.append(self.MSE(target, prediction, is_test = True))
            if test_data is not None:
                self.test_loss.append(self.MSE(test_target, self.predict(test_data), is_test = True))

            
    def predict(self, data): 
        ''' Prediction '''
        y = None  
        for ind, estimator in enumerate(self.estimators_list):
            if y is not None:
                y += estimator.predict(data) * self.b[ind] 
            else:
                y = estimator.predict(data) 
        return y