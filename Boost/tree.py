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

class Node(object):
    ''' A node of a Tree '''

    def __init__(self, mapping, max_depth):

        self.mapping = mapping
        self.max_depth = max_depth

        self.left = None
        self.right = None

        self.ind = None
        self.value = None

        self.finished_value = None
         
        
    def _decision_indexes(self, data, ind = None, value = None):
        '''
        Count the decision mask for the data
        (left node - True, right node - False)

        Parameters:
            * data (np.array)
            * ind (int or None)
              Default: None
            * value (int or None)
              Default: None

        Returns:
            * np.array mask of True/False values 
        '''
        if ind is None:
            ind = self.ind
            value = self.value

        return data[:, ind] <= value

        
    def predict(self, data):
        ''' Prediction '''
        
        target = np.zeros(len(data))
        
        # if this is the last node of the branch
        if self.finished_value is not None:
            target += self.finished_value
        else:
            mapping = self._decision_indexes(data)
            target[mapping] = self.left.predict(data[mapping])
            target[mapping == False] = self.right.predict(data[mapping == False])

        return target


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
            predicted = np.mean(target)
        if is_mean:
            MSE = ((predicted - target)**2).sum() 
        else:
            MSE = ((predicted - target)**2).sum() / float(len(target))
        return MSE

        
    def find_best_split_on_feature(self, data, target, ind, mask):
        '''
        Find the best split of the data for the index ind

        Main Idea:
        S = MSE
        max(S - S1 * N1/N - S2 * N2/N)
        but S is fixed and N is fixed 
        so min(S1 * N1 + S2 * N2)

        Parameters:
            * data (np.array)
            * target (np.array)
            * ind (int)
            * mask (np.array with data shape)


        Returns:
            * best value 
            (None if there is no best split)
            * min loss
            (None if there is no best split)

        '''
        
        best = None
        prev_value = None        
        min_loss = sys.maxsize
        
        
        for value in data[mask][self.mapping[mask]][:, ind]:
            
            # We don't have to check repeats
            if value == prev_value:
                continue
                
            left_ind = self._decision_indexes(data, ind, value)
            
            # if everything in one node - so it is the end
            if left_ind[self.mapping].all():
                break
            
            #count mapped targets
            left_target = target[self.mapping][left_ind[self.mapping]]
            right_target = target[self.mapping][left_ind[self.mapping] == False]
            
            left_loss  = self.MSE(left_target) * len(left_target)
            right_loss = self.MSE(right_target) * len(right_target)

            loss = left_loss + right_loss
            
            if loss < min_loss:
                min_loss = loss
                best = value
                
            prev_value = value
        
                
        return best, min_loss
    
    
    
    def fit(self, data, target, mask):
        '''
        Fitting model

        Parameters:
            * data (np.array)
            * target (np.array)
            * mask (np.array with data shape) - argsort for every feature of data 

        Returns:
            * self
        '''

        # if there are targets of only one class
        if len(np.unique(target[self.mapping])) < 2 or self.max_depth < 1:
            self.finished_value = np.mean(target[self.mapping])
            return self
        
        
        # lets count the min loss for each feature
        losses = []
        values = []

        for ind in range(data.shape[1]):  

            best_values, loss = self.find_best_split_on_feature(data, target, ind, mask[ind])

            # if there is no best split
            if best_values is None:
                losses.append(sys.maxsize)
            else:
                losses.append(loss)
            
            values.append(best_values)
        
        # Find the ind of feature with the min loss and it's value
        self.ind = np.argmin(losses)
        self.value = values[self.ind]
        
        # if we can't devide the data
        if self.value is None:
            print ("OOOOOOOPS: self.vaue is None!")
            self.finished_value = np.mean(target[self.mapping])
        
        else:

            mapping = self._decision_indexes(data)
            self.left = Node(mapping * self.mapping, self.max_depth - 1).fit(data, target, mask)
            self.right = Node((mapping == False) * self.mapping, self.max_depth - 1).fit(data, target, mask)

        return self
    


class Tree(object):
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


    @staticmethod
    def _create_mask(data):
        ''' Creating argsort mask for each feature in data '''
        mask = np.zeros((data.shape[1], data.shape[0]), dtype=int)
        for ind,_ in enumerate(data[0,:]):
            mask[ind] = np.argsort(data[:,ind])
        return mask        
    

    def fit(self, data, target):
        ''' Fitting model '''

        self.Node = Node(np.ones(len(target)) == 1, self.max_depth)
        sort_mask = self._create_mask(data)

        self.Node.fit(data, target, sort_mask)
        return self


    def predict(self, data):
        ''' Prediction '''
        return self.Node.predict(data)