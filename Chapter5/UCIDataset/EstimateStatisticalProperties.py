# Author: Matthew Wicker

# Alright, hold on to your socks this one might be more rough...
import os
import sys
import logging
sys.path.append("../../")
import deepbayesHF
from deepbayesHF import PosteriorModel
from deepbayesHF.analyzers import IBP_prob
from deepbayesHF.analyzers import IBP_upper
from deepbayesHF.analyzers import FGSM
from deepbayesHF.analyzers import massart_bound_check

from datasets import Dataset
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--imnum")
parser.add_argument("--dataset")
parser.add_argument("--epsilon", required=False, default=0.025)

args = parser.parse_args()
imnum = int(args.imnum)
epsilon = float(args.epsilon)
dataset_string = str(args.dataset)
INDEX = imnum

data = Dataset(dataset_string)
from sklearn.preprocessing import StandardScaler

X_train = np.asarray(data.train_set.train_data)
y_train = np.asarray(data.train_set.train_labels.reshape(-1,1))

X_test = np.asarray(data.test_set.test_data)
y_test = np.asarray(data.test_set.test_labels.reshape(-1,1))

X_scaler = StandardScaler()
X_scaler.fit(X_train)
X_train, X_test = X_scaler.transform(X_train), X_scaler.transform(X_test)

# If we are running with scaled posteriors
y_scaler = StandardScaler()
y_scaler.fit(y_train)
y_train, y_test = y_scaler.transform(y_train), y_scaler.transform(y_test)

print("Sucessfully loaded dataset %s \t \t with train shapes %s, %s"%(dataset_string, X_train.shape, y_train.shape))

input_range = np.max(X_test, axis=0) - np.min(X_test, axis=0)
input_range /= 2
print("Input range: ", input_range)
output_range = np.max(y_test) - np.min(y_test)
print("Output range: ", np.max(y_test) - np.min(y_test))
#DELTA = 0.05 * output_range
DELTA = 0.075 * output_range
print("HERE IS DELTA: ", DELTA)
# Dataset information:
in_dims = X_train.shape[1]

import numpy as np
# Load in approximate posterior distribution
bayes_model = PosteriorModel("PosteriorScaled/VOGN_%s_Model"%(dataset_string))

# Lets define our correctness property (e.g. correct class)
TRUE_VALUE = bayes_model.predict(np.asarray([X_test[INDEX]])).numpy() #y_test[INDEX]
EPSILON = epsilon #0.01 #input_range * 0.01


def predicate_safe(iml, imu, ol, ou):
    bound_above = TRUE_VALUE + DELTA
    bound_below = TRUE_VALUE - DELTA
    if(ol >= bound_below and ou <= bound_above):
        return True
    else:
        return False

def predicate_worst(worst_case):
    if(worst_case < TRUE_VALUE - DELTA or worst_case > TRUE_VALUE + DELTA):
        print(worst_case, TRUE_VALUE)
        return False
    else:
        return True

# We should do clipping, but its 4am now and I am lazy
img = np.asarray([X_test[INDEX]])
img_upper = np.asarray([X_test[INDEX]+(input_range*EPSILON)])
img_lower = np.asarray([X_test[INDEX]-(input_range*EPSILON)])

# LOAD A MODEL
model = bayes_model

# We have the input, output, specification, and network now so we would like to:

# FILE: %dataset_chernoff.log
# 1a: Estimate with chernoff bound with attacks
# 1b: Estimate with chernoff bound with bound propagation


# INFO NEEDED - sample #, label, prediction, epsilon, p_safe, d_safe, b_eps, b_delta, numsamples, time


#def massart_bound_check(model, inp, eps, predicate, **kwargs):
#    delta = kwargs.get('delta', 0.3)
#    alpha = kwargs.get('alpha', 0.05)
#    confidence = kwargs.get('confidence', 0.95)
#    verbose = kwargs.get('verbose', False)
#    verify = kwargs.get('verify', True)
import time
dir = "ReduxLogs"

pred = model.predict(img)

#img = np.asarray([img])

start = time.process_time()
p_safe_attack, iterations_attack, mean = massart_bound_check(model, img, EPSILON, predicate_worst, cls=TRUE_VALUE,
                                                             confidence=0.9, delta=0.1, alpha=0.05, 
                                                             verify=False, chernoff=True, verbose=True)
attk_time = time.process_time() - start
d_safe_attack = predicate_worst(mean)

start = time.process_time()
p_safe_bounds, iterations_bounds, mean = massart_bound_check(model, img, EPSILON, predicate_worst, cls=TRUE_VALUE,
                                                             confidence=0.9, delta=0.1, alpha=0.05, 
                                                             verify=True, chernoff=True, verbose=True)
veri_time = time.process_time() - start
d_safe_bounds= predicate_worst(mean)
#print("1_%s_2_%s_3_%s_4_%s_5_%s_6_%s_7_%s_8_%s_9_%s_10_%s_11_%s_12_%s_13_%s_14_%s_15_%s"%(INDEX, 
#      TRUE_VALUE, pred, EPSILON, p_safe_attack, d_safe_attack, p_safe_bounds, d_safe_bounds, 0.05, 0.1, 0.05, iterations_attack, iterations_bounds, veri_time, attk_time))

import json
with open("%s/%s_chernoff.log"%(dir, dataset_string), 'a') as f:
    record = {'index': INDEX, 'label':float(TRUE_VALUE[0]), 'pred':float(pred.numpy()[0][0]), 'epsilon':EPSILON, 
              'p_safe_attack':p_safe_attack, 'd_safe_attack':d_safe_attack, 'p_safe_bounds':p_safe_bounds, 'd_safe_bounds':d_safe_bounds, 
              'beps':0.05, 'bdelt':0.1, 'balpha':0.05, 'iterations_attack':iterations_attack, 'iterations_bounds':iterations_bounds, 'veri_time':veri_time, 'attk_time':attk_time }
    json.dump(record, f)
    f.write(os.linesep)
#logging.basicConfig(filename="ExperimentalLogs/%s_chernoff.log"%(dataset_string),level=logging.DEBUG)
#logging.info("i#_%s_eps_%s\n"%(INDEX, EPSILON))
#logging.info("i#_%s_eps_%s_p_%s"%(INDEX, max_eps_veri, p_of_veri))
#logging.info("1_%s_2_%s_3_%s_4_%s_5_%s_6_%s_7_%s_8_%s_9_%s_10_%s_11_%s_12_%s_13_%s_14_%s_15_%s"%(INDEX, 
#      TRUE_VALUE, pred, EPSILON, p_safe_attack, d_safe_attack, p_safe_bounds, d_safe_bounds, 0.05, 0.1, 0.05, iterations_attack, iterations_bounds, veri_time, attk_time))




start = time.process_time()
p_safe_attack, iterations_attack, mean = massart_bound_check(model, img, EPSILON, predicate_worst, cls=TRUE_VALUE,
                                                             confidence=0.9, delta=0.1, alpha=0.05, 
                                                             verify=False, chernoff=False, verbose=True)
attk_time = time.process_time() - start
d_safe_attack = predicate_worst(mean)

start = time.process_time()
p_safe_bounds, iterations_bounds, mean = massart_bound_check(model, img, EPSILON, predicate_worst, cls=TRUE_VALUE,
                                                             confidence=0.9, delta=0.1, alpha=0.05, 
                                                             verify=True, chernoff=False, verbose=True)
veri_time = time.process_time() - start
d_safe_bounds= predicate_worst(mean)

import json
with open("%s/%s_massart.log"%(dir, dataset_string), 'a') as f:
    record = {'index': INDEX, 'label':float(TRUE_VALUE[0]), 'pred':float(pred.numpy()[0][0]), 'epsilon':EPSILON, 
              'p_safe_attack':p_safe_attack, 'd_safe_attack':d_safe_attack, 'p_safe_bounds':p_safe_bounds, 'd_safe_bounds':d_safe_bounds, 
              'beps':0.05, 'bdelt':0.1, 'balpha':0.05, 'iterations_attack':iterations_attack, 'iterations_bounds':iterations_bounds, 'veri_time':veri_time, 'attk_time':attk_time }
    json.dump(record, f)
    f.write(os.linesep)

"""
import logging
logging.basicConfig(filename="ExperimentalLogs/%s_massart.log"%(dataset_string),level=logging.DEBUG)
print("1_%s_2_%s_3_%s_4_%s_5_%s_6_%s_7_%s_8_%s_9_%s_10_%s_11_%s_12_%s_13_%s_14_%s_15_%s"%(INDEX, 
      TRUE_VALUE, pred, EPSILON, p_safe_attack, d_safe_attack, p_safe_bounds, d_safe_bounds, 0.05, 0.1, 0.05, iterations_attack, iterations_bounds, veri_time, attk_time))
#logging.info("i#_%s_eps_%s\n"%(INDEX, EPSILON))
#logging.info("i#_%s_eps_%s_p_%s"%(INDEX, max_eps_veri, p_of_veri))
logging.info("1_%s_2_%s_3_%s_4_%s_5_%s_6_%s_7_%s_8_%s_9_%s_10_%s_11_%s_12_%s_13_%s_14_%s_15_%s"%(INDEX, 
      TRUE_VALUE, pred, EPSILON, p_safe_attack, d_safe_attack, p_safe_bounds, d_safe_bounds, 0.05, 0.1, 0.05, iterations_attack, iterations_bounds, veri_time, attk_time))
"""

