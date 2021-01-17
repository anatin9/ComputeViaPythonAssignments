#!/usr/bin/python

#####################################
## module: leave_one_out_cross_validation.py
## Austin Coelho
#####################################

from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import LeaveOneOut

# load the iris dataset
data = load_iris()
flowers = data.data ## 150x4 array flower feature vectors
feature_names = data.feature_names ## names of features 
target = data.target ## target numbers
target_names = data.target_names ## target names
 
## Get an array of flower names
flower_names = target_names[target]

# Build three boolean indexes to retrieve
# setosas, virginicas, and versicolors from flowers, e.g.
is_setosa     = (flower_names == 'setosa')
is_virginica  = (flower_names == 'virginica')
is_versicolor = (flower_names == 'versicolor')

def compute_model_accuracy(predictions, ground_truth):
    return float(np.sum(predictions == ground_truth))/len(ground_truth)
    

def run_model(model, flowers):
    trutharr = np.array([])
    for x in flowers:
	if x[model[0]] > model[1] and not model[2]:
		trutharr = np.append(trutharr, True)
	elif x[model[0]] <= model[1] and model[2]:
		trutharr = np.append(trutharr, True)
	else:
		trutharr = np.append(trutharr, False)
    return trutharr
   
    
def learn_best_th_model_for(flower_name, flowers, bool_index):
    assert len(flowers) == len(bool_index)
    bestfn = 0
    bestth = 0
    best_reverse = False
    bestacc = -1.0
    if flower_name == 'versicolor' or flower_name == 'virginica':
	is_setosa = (flower_names ==  'setosa')
	non_setosa = flowers[~is_setosa]
	non_setosa_names = flower_names[~is_setosa]
	is_virginica = (non_setosa_names ==  'virginica')
	is_versicolor = (non_setosa_names == 'versicolor')
	for fn in xrange(non_setosa.shape[1]):
		if flower_name == 'virginica':
			possible_threshold = non_setosa[:, fn]
			for pt in possible_threshold:
				feature_vals = non_setosa[:,fn]
				predictions = (feature_vals > pt)
				acc = (predictions == is_virginica).mean()
				revacc = (predictions == ~is_virginica).mean()
				if revacc > acc:
				    reverse = True
				    acc = revacc
			        else:
				    reverse = False

    			        if acc > bestacc:  
				    bestacc = acc
				    bestfn = fn
				    bestth = pt
				    best_reverse = reverse
		if flower_name == 'versicolor':
			possible_threshold = non_setosa[:, fn]
			for pt in possible_threshold:
				feature_vals = non_setosa[:,fn]
				predictions = (feature_vals > pt)
				acc = (predictions == is_versicolor).mean()
				revacc = (predictions == ~is_versicolor).mean()
				if revacc > acc:
				    reverse = True
				    acc = revacc
				else:
				    reverse = False

				if acc > bestacc:  
				    bestacc = acc
				    bestfn = fn
				    bestth = pt
				    best_reverse = reverse

				
    if flower_name == 'setosa':
	is_virginica  = (flower_names == 'virginica')
	non_virginica = flowers[~is_virginica]
	non_virginica_names = flower_names[~is_virginica]
	is_setosa = (non_virginica_names ==  'setosa')
	for fn in xrange(non_virginica.shape[1]):
			possible_threshold = non_virginica[:, fn]
			for pt in possible_threshold:
				feature_vals = non_virginica[:,fn]
				predictions = (feature_vals > pt)
				acc = (predictions == is_setosa).mean()
				revacc = (predictions == ~is_setosa).mean()
				if revacc > acc:
				    reverse = True
				    acc = revacc
			        else:
				    reverse = False

    			        if acc > bestacc:  
				    bestacc = acc
				    bestfn = fn
				    bestth = pt
				    best_reverse = reverse
    return(bestfn, bestth, best_reverse, bestacc)


	
            

def leave_one_out_cross_validation(flower_name, flowers):
     if flower_name == 'setosa':
	model = learn_best_th_model_for('setosa', flowers, is_setosa)
    	predictions = run_model(model, flowers)
	predictions = np.delete(predictions, 1)
	return compute_model_accuracy(predictions, np.delete(is_setosa, 1))
     if flower_name == 'virginica':
	model = learn_best_th_model_for('virginica', flowers, is_virginica)
    	predictions = run_model(model, flowers)
	predictions = np.delete(predictions, 1)
	return compute_model_accuracy(predictions, np.delete(is_virginica, 1))
     if flower_name == 'versicolor':
	model = learn_best_th_model_for('versicolor', flowers, is_versicolor)
   	predictions = run_model(model, flowers)
	predictions = np.delete(predictions, 1)
	return compute_model_accuracy(predictions, np.delete(is_versicolor, 1))


# ---------------- UNIT TESTS ------------------------

def unit_test_01():
    '''learn single feature classifier for setosa'''
    setosa_model = learn_best_th_model_for('setosa', flowers,
                                           is_setosa)
    print 'setosa model:', setosa_model

def unit_test_02():
    '''learn single feature classifier for virginica'''
    virginica_model = learn_best_th_model_for('virginica', flowers,
                                              is_virginica)
    print 'virginica model:', virginica_model

def unit_test_03():
    '''learn single feature classifier for versicolor'''
    versicolor_model = learn_best_th_model_for('versicolor', flowers,
                                               is_versicolor)
    print 'versicolor model:', versicolor_model

def unit_test_04():
    '''learn and run single feature classifier for setosa'''
    model = learn_best_th_model_for('setosa', flowers, is_setosa)
    predictions = run_model(model, flowers)
    print 'setosa model acc:', compute_model_accuracy(predictions, is_setosa)

def unit_test_05():
    '''learn and run single feature classifier for virginica'''
    model = learn_best_th_model_for('virginica', flowers, is_virginica)
    predictions = run_model(model, flowers)
    print 'virginica model acc:', compute_model_accuracy(predictions, is_virginica)

def unit_test_06():
    '''learn and run single feature classifier for versicolor'''
    model = learn_best_th_model_for('versicolor', flowers, is_versicolor)
    predictions = run_model(model, flowers)
    print 'versicolor model acc:', compute_model_accuracy(predictions, is_versicolor)

def unit_test_07():
    '''run leave-one-out cross-validation for setosas'''
    acc = leave_one_out_cross_validation('setosa', flowers)
    print 'leave-1-out cross_valid acc for setosa:', acc

def unit_test_08():
    '''run leave-one-out cross-validation for virginicas'''
    acc = leave_one_out_cross_validation('virginica', flowers)
    print 'leave-1-out cross_valid acc for virginica:', acc  

def unit_test_09():
    '''run leave-one-out cross-validation for versicolors'''
    acc = leave_one_out_cross_validation('versicolor', flowers)
    print 'leave-1-out cross_valid acc for versicolor:', acc
    

if __name__ == '__main__':
     unit_test_01()
     unit_test_02()
     unit_test_03()
     unit_test_04()
     unit_test_05()
     unit_test_06()
     unit_test_07()
     unit_test_08()
     unit_test_09()
     pass
