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

import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--imnum")
parser.add_argument("--infer")

args = parser.parse_args()
imnum = int(args.imnum)
post_string = str(args.infer)
INDEX = imnum
EPSILON = 0.01
MARGIN = 2.5
SAMPLES = 750
# LOAD IN THE DATA

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

# SELECT THE INPUT
img = np.asarray([X_test[INDEX]])
TRUE_VALUE = y_test[INDEX]

import numpy as np
# Load in approximate posterior distribution
bayes_model = PosteriorModel("Posteriors/%s_FCN_Posterior_1"%(post_string))

img = np.asarray([X_test[INDEX]])
img_upper = np.clip(np.asarray([X_test[INDEX]+(EPSILON)]), 0, 1)
img_lower = np.clip(np.asarray([X_test[INDEX]-(EPSILON)]), 0, 1)


# We start with epsilon = 0.0 and increase it as we go.
p_lower, _ = IBP_prob(bayes_model, img_lower, img_upper, MARGIN, SAMPLES, predicate=predicate_safe, inflate=1.0)
print("Initial Safety Probability: ", p_lower)
dir = "ExperimentalLogs"
import json
iterations = 4
max_eps_veri = 0.0
p_of_veri = 0.0

record = {"Index":INDEX, "Lower":p_lower, "Samples":SAMPLES, "Margin":MARGIN, "Epsilon":EPSILON,  "Iterations":iterations}
with open("%s/%s_lower.log"%(dir, post_string), 'a') as f:
    json.dump(record, f)
    f.write(os.linesep)


stepsize = 0.10
max_eps_veri = 0.0
p_of_veri = 0.0
for i in range(2):
    if(p_lower >= 0.8):
        max_eps_veri = max(EPSILON, max_eps_veri)
        p_of_veri = p_lower #max(EPSILON, max_eps_veri)
        EPSILON *= 1.5
        img_upper = np.clip(np.asarray([X_test[INDEX]+(EPSILON)]), 0, 1)
        img_lower = np.clip(np.asarray([X_test[INDEX]-(EPSILON)]), 0, 1)
    else:
        EPSILON /= 2
        img_upper = np.clip(np.asarray([X_test[INDEX]+(EPSILON)]), 0, 1)
        img_lower = np.clip(np.asarray([X_test[INDEX]-(EPSILON)]), 0, 1)
    print("Computing with epsilon: ", EPSILON)
    print("Stepsize: ", stepsize)
    p_lower, _ = IBP_prob(bayes_model, img_lower, img_upper, MARGIN, SAMPLES, predicate=predicate_safe, inflate=1.0)
    print("Probability: ", p_lower)
    record = {"Index":INDEX, "Lower":p_lower, "Samples":SAMPLES, "Margin":MARGIN, "Epsilon":EPSILON, "Iterations":iterations}
    with open("%s/%s_lower.log"%(dir, post_string), 'a') as f:
        json.dump(record, f)
        f.write(os.linesep)

"""
import logging
logging.basicConfig(filename="MSELogs/%s_eps.log"%(dataset_string),level=logging.DEBUG)
print("i#_%s_eps_%s_p_%s"%(INDEX, max_eps_veri, p_of_veri))
#logging.info("i#_%s_eps_%s\n"%(INDEX, EPSILON))
logging.info("i#_%s_eps_%s_p_%s"%(INDEX, max_eps_veri, p_of_veri))
"""


