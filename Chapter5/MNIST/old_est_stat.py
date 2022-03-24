# Estimate Statistical Properties

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

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--imnum")
parser.add_argument("--infer")
parser.add_argument("--epsilon", required=False, default=0.05)

args = parser.parse_args()
epsilon = float(args.epsilon)
imnum = int(args.imnum)
post_string = str(args.infer)
INDEX = imnum

EPSILON = epsilon
#DELTA - CLASS CONSISTENCY

# LOAD IN THE DATA

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train/255.
X_test = X_test/255.
X_train = X_train.astype("float64").reshape(-1, 28*28)
X_test = X_test.astype("float64").reshape(-1, 28* 28)

# SELECT THE INPUT
img = np.asarray([X_test[INDEX]])
TRUE_VALUE = y_test[INDEX]

# SET UP THE PROPERTY

def predicate_worst(worst_case):
    if(np.argmax(worst_case) != TRUE_VALUE):
        return False
    else:
        return True

# LOAD IN THE MODEL
model = PosteriorModel('Posteriors/%s_FCN_Posterior_0'%(post_string))

# TEST THE MODEL AND LOG THE RESULT

import time
dir = "ReduxLogs"
pred = model.predict(img)

start = time.process_time()
p_safe_attack, iterations_attack, mean = massart_bound_check(model, img, EPSILON, predicate_worst, cls=TRUE_VALUE,
                                                             confidence=0.9, delta=0.1, alpha=0.05, classification=True,
                                                             verify=False, chernoff=True, verbose=True)
attk_time = time.process_time() - start
d_safe_attack = predicate_worst(mean)

start = time.process_time()
p_safe_bounds, iterations_bounds, mean = massart_bound_check(model, img, EPSILON, predicate_worst, cls=TRUE_VALUE,
                                                             confidence=0.9, delta=0.1, alpha=0.05, classification=True,
                                                             verify=True, chernoff=True, verbose=True)
veri_time = time.process_time() - start
d_safe_bounds= predicate_worst(mean)

print(pred)
import json
with open("%s/%s_chernoff.log"%(dir, post_string), 'a') as f:
    record = {'index': INDEX, 'label':float(TRUE_VALUE), 'pred':float(np.argmax(pred.numpy())), 'epsilon':EPSILON, 
              'p_safe_attack':p_safe_attack, 'd_safe_attack':d_safe_attack, 'p_safe_bounds':p_safe_bounds, 'd_safe_bounds':d_safe_bounds, 
              'beps':0.05, 'bdelt':0.1, 'balpha':0.05, 'iterations_attack':iterations_attack, 'iterations_bounds':iterations_bounds, 'veri_time':veri_time, 'attk_time':attk_time }
    json.dump(record, f)
    f.write(os.linesep)



start = time.process_time()
p_safe_attack, iterations_attack, mean = massart_bound_check(model, img, EPSILON, predicate_worst, cls=TRUE_VALUE,
                                                             confidence=0.9, delta=0.1, alpha=0.05, classification=True,
                                                             verify=False, chernoff=False, verbose=True)
attk_time = time.process_time() - start
d_safe_attack = predicate_worst(mean)

start = time.process_time()
p_safe_bounds, iterations_bounds, mean = massart_bound_check(model, img, EPSILON, predicate_worst, cls=TRUE_VALUE,
                                                             confidence=0.9, delta=0.1, alpha=0.05, classification=True,
                                                             verify=True, chernoff=False, verbose=True)
veri_time = time.process_time() - start
d_safe_bounds= predicate_worst(mean)

import json
with open("%s/%s_massart.log"%(dir, post_string), 'a') as f:
    record = {'index': INDEX, 'label':float(TRUE_VALUE), 'pred':float(np.argmax(pred.numpy())), 'epsilon':EPSILON, 
              'p_safe_attack':p_safe_attack, 'd_safe_attack':d_safe_attack, 'p_safe_bounds':p_safe_bounds, 'd_safe_bounds':d_safe_bounds, 
              'beps':0.05, 'bdelt':0.1, 'balpha':0.05, 'iterations_attack':iterations_attack, 'iterations_bounds':iterations_bounds, 'veri_time':veri_time, 'attk_time':attk_time }
    json.dump(record, f)
    f.write(os.linesep)





