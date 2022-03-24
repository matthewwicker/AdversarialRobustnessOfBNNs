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

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

#tf.debugging.set_log_device_placement(True)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--imnum")
parser.add_argument("--infer")
parser.add_argument("--width", required=False, default=1)
parser.add_argument("--depth", required=False, default=1)
parser.add_argument("--epsilon", required=False, default=0.05)
parser.add_argument("--cls", default=0)
parser.add_argument("--red", default=0.0)

args = parser.parse_args()
imnum = int(args.imnum)
width = int(args.width)
depth = int(args.depth)
EPSILON = float(args.epsilon)
cls = int(args.cls)
red = float(args.red)
post_string = str(args.infer)
INDEX = imnum
MARGIN = 2.5
SAMPLES = 750

# 2.5, 750
# LOAD IN THE DATA
print("Finished parsing args")

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train/255.
X_test = X_test/255.
X_train = X_train.astype("float64").reshape(-1, 28*28)
X_test = X_test.astype("float64").reshape(-1, 28* 28)

def predicate_safe(iml, imu, ol, ou):
    v1 = tf.one_hot(TRUE_VALUE, depth=10)
    v2 = 1 - tf.one_hot(TRUE_VALUE, depth=10)
    v1 = tf.squeeze(v1); v2 = tf.squeeze(v2)
    worst_case = tf.math.add(tf.math.multiply(v2, ou), tf.math.multiply(v1, ol))
    if(np.argmax(worst_case) == TRUE_VALUE):
        return True
    else:
        return False

def predicate_worst(worst_case):
    if(np.argmax(worst_case) != TRUE_VALUE):
        return False
    else:
        return True

print("Finished loading data")


import numpy as np
# Load in approximate posterior distribution
bayes_model = PosteriorModel("ReducedPosteriors/%s_FCN_Posterior_%s_%s"%(post_string, cls, red))

bayes_model.posterior_var += 0.000000001
model = bayes_model

#class_values = np.argwhere(y_test == cls)
#INDEX = class_values[INDEX]
#INDEX = np.squeeze(INDEX)
#INDEX = int(INDEX)

print("Testing with index: ", INDEX)
# SELECT THE INPUT
img = np.asarray([X_test[INDEX]])
TRUE_VALUE = y_test[INDEX]
#TRUE_VALUE = np.argmax(bayes_model.predict(img)) #y_test[INDEX]


img = np.asarray([X_test[INDEX]])
img_upper = np.clip(np.asarray([X_test[INDEX]+(EPSILON)]), 0, 1)
img_lower = np.clip(np.asarray([X_test[INDEX]-(EPSILON)]), 0, 1)


# TEST THE MODEL AND LOG THE RESULT

import time
dir = "MaxEps"
pred = model.predict(img)

print("Finished setting up tests")


start = time.process_time()
#p_safe_attack, iterations_attack, mean = massart_bound_check(model, img, EPSILON, predicate_worst, cls=TRUE_VALUE,
#                                                             confidence=0.9, delta=0.1, alpha=0.05, classification=True,
#                                                             verify=False, chernoff=True, verbose=True)
attk_time = time.process_time() - start
#d_safe_attack = predicate_worst(mean)
p_safe_attack, d_safe_attack = -1, -1
iterations_attack = -1
start = time.process_time()

MAXEPS = 0.0
for eps in np.linspace(0, 0.05, 21):
    p_safe_bounds, iterations_bounds, mean = massart_bound_check(model, img, eps, predicate_worst, cls=TRUE_VALUE,
                                                             confidence=0.85, delta=0.15, alpha=0.05, classification=True,
                                                             verify=True, chernoff=False, verbose=True)
    if(p_safe_bounds > 0.75):
        MAXEPS = eps
    else:
        break

veri_time = time.process_time() - start
d_safe_bounds= predicate_worst(mean)


print(pred)
print("~~~~~~~~~~~~~~~~ WRITING OUT TO FILE NOW ~~~~~~~~~~~~~~~~")
import json
with open("%s/%s_maxeps.log"%(dir, post_string), 'a') as f:
    record = {'index': INDEX, 'label':float(TRUE_VALUE), 'pred':float(np.argmax(pred.numpy())), 'epsilon':MAXEPS, 'width':width, 'depth':depth,
              'p_safe_attack':p_safe_attack, 'd_safe_attack':d_safe_attack, 'p_safe_bounds':p_safe_bounds, 'd_safe_bounds':d_safe_bounds, 
              'beps':0.05, 'bdelt':0.1, 'balpha':0.05, 'iterations_attack':iterations_attack, 'iterations_bounds':iterations_bounds, 'veri_time':veri_time, 'attk_time':attk_time,  'reduced':red}
    print(record)
    json.dump(record, f)
    f.write(os.linesep)





