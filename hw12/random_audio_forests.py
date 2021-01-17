#!/usr/bin/python

'''
========================================
module: random_audio_forests.py
Austin Coelho
========================================
'''

import numpy as np
import scipy
import random
import os
import glob
import argparse

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import KFold
import matplotlib.colors
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


base_dir = '/home/austin/Downloads/hw12/audio_data/'
sub_dir = ['beefv','cricketfv','noisefv']

## Reading the files and creating feature and response object
def read_audio_data(base_dir):
    global sub_dir
    data, target = [], []
    for label,class_names in enumerate(sub_dir, start = 0):
        vector_dir = os.path.join(base_dir, class_names, '*.mvector.npy')
        all_files = glob.glob(vector_dir)
        for f in all_files :
            value = np.load(f)
            data.append(value[:])
            target.append(label)

    return np.array(data), np.array(target)

def train_test_split_eval_dtr(dtr, data, target, test_size=0.4):
	trainData, testData, trainTarget, testTarget = train_test_split(data, target, test_size=test_size, random_state = random.randint(0,1000))
	dt = dtr.fit(trainData, trainTarget)
	test_len = float(len(testTarget))
	accPred = sum(dt.predict(testData) == testTarget)
	print ("Dtr accuracy = " + str(accPred/test_len))
	clfExpected = testTarget
	clfPredicted = dtr.predict(testData)
	cm = confusion_matrix(clfExpected, clfPredicted)
	print "Confusion matrix of dtr"
	print cm
	print ("_____________________________________________________________")

def train_test_split_eval_rf(rf, data, target, test_size=0.4):
	trainData, testData, trainTarget, testTarget = train_test_split(data, target, test_size=test_size, random_state = random.randint(0,1000))
	rfd = rf.fit(trainData, trainTarget)
	testLen = float(len(testTarget))
	accPred = sum(rfd.predict(testData) == testTarget)
	print ("# of trees in random forest: "+ str(len(rf)))
	print("Rf accuracy = " +str(accPred/testLen))
	clfExp = testTarget
	clfPred = rf.predict(testData)
	cm = confusion_matrix(clfExp, clfPred)
	print "Confusion matrix of rf"
	print cm
	print ("_____________________________________________________________")

def train_test_split_dtr_range_eval(n, data, target):
	for x in xrange(0, n):
		dtr = tree.DecisionTreeClassifier(random_state = random.randint(0, 100))	
		train_test_split_eval_dtr(dtr, data, target)


def train_test_split_rf_range_eval(lower_tree_bound, upper_tree_bound, data, target):
	for x in xrange(lower_tree_bound, upper_tree_bound+10, 10):
		rf = RandomForestClassifier(x, random_state=random.randint(0, 1000))
		train_test_split_eval_rf(rf, data, target)

if __name__ == '__main__':
   import argparse
   import time
   ap = argparse.ArgumentParser()
   ap.add_argument('-root_dir', '--root_dir', help='root directory', required=True)
   args = vars(ap.parse_args())
   start_time = time.time()
   base_dir = args['root_dir']
   data, target = read_audio_data(base_dir)
   







