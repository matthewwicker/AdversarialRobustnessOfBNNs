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
MARGIN = 0.5
SAMPLES = 500
EPSILON = 0.05

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

# LOAD IN THE MODEL
bayes_model = PosteriorModel('Posteriors/%s_FCN_Posterior_0'%(post_string))

# Lets define our correctness property (e.g. correct class)
TRUE_VALUE = y_test[INDEX]

def predicate_unsafe(iml, imu, ol, ou):
    v1 = tf.one_hot(TRUE_VALUE, depth=10)
    v2 = 1 - tf.one_hot(TRUE_VALUE, depth=10)
    v1 = tf.squeeze(v1); v2 = tf.squeeze(v2)
    best_case = tf.math.add(tf.math.multiply(v1, ou), tf.math.multiply(v2, ol))
    if(np.argmax(best_case) != TRUE_VALUE):
        return True
    else:
        return False

stepsize = 0.25
EPSILON = 2.5
min_eps_veri = 1.0
p_of_veri = 1.0

# Generate a series of attacks
from deepbayesHF import analyzers
meth = analyzers.PGD
loss = tf.keras.losses.MeanSquaredError()

attacks = []
img = np.asarray([X_test[INDEX]])
img_upper = np.clip(np.asarray([X_test[INDEX]+(EPSILON)]), 0, 1) 
img_lower = np.clip(np.asarray([X_test[INDEX]-(EPSILON)]), 0, 1) 
adv = analyzers.FGSM(bayes_model, np.asarray([X_test[INDEX]]), eps=EPSILON, loss_fn=loss)
attacks.append(adv)
p_upper, _ = IBP_upper(bayes_model, img, img, MARGIN, SAMPLES, loss_fn=loss, eps=EPSILON, predicate=predicate_unsafe, inputs=attacks, inflate=2.5, mod_option=10)
p_upper = 1 - p_upper

print("Initial Unsafe Probability: ", p_upper)

#if(p_upper > 0.1):
#    import logging
#    logging.basicConfig(filename="MURLogs/%s_upper_eps.log"%(dataset_string),level=logging.DEBUG)
#    #print("i#_%s_eps_%s_p_%s"%(INDEX, max_eps_veri, p_of_veri))
#    logging.info("i#_%s_eps_%s_p_%s"%(INDEX, 1.0, 1.0))
#    sys.exit()
iterations = 4
for i in range(iterations):
    if(p_upper <= 0.1):
        EPSILON -= stepsize
        stepsize /= 2
        min_eps_veri = min(EPSILON, min_eps_veri)
        p_of_veri = p_upper
        #img_upper = np.asarray([X_test[INDEX]+(input_range*EPSILON)])
        #img_lower = np.asarray([X_test[INDEX]-(input_range*EPSILON)])
    else:
        EPSILON += stepsize
        stepsize /= 2
        #img_upper = np.asarray([X_test[INDEX]+(input_range*EPSILON)])
        #img_lower = np.asarray([X_test[INDEX]-(input_range*EPSILON)])
    print("Computing with epsilon: ", EPSILON)
    print("Stepsize: ", stepsize)
    attacks = []
    adv = analyzers.FGSM(bayes_model, np.asarray([X_test[INDEX]]), eps=EPSILON, loss_fn=loss)
    attacks.append(adv)
    p_upper, _ = IBP_upper(bayes_model, img, img, MARGIN, SAMPLES, loss_fn=loss, eps=EPSILON, predicate=predicate_unsafe, inputs=attacks, inflate=2.5, mod_option=3)
    p_upper = 1 - p_upper
    print("Probability: ", p_upper)

dir = "ExperimentalLogs"
record = {"Index":INDEX, "Upper":p_upper, "Samples":Samples, "Margin":MARGIN, "Epsilon":EPSILON, "Stepsize":stepsize, "Iterations":iterations}
import json
with open("%s/%s_upper.log"%(dir, post_string), 'a') as f:
    json.dump(record, f)
    f.write(os.linesep)
"""
import logging
logging.basicConfig(filename="MURLogs/%s_upper_eps.log"%(dataset_string),level=logging.DEBUG)
print("i#_%s_eps_%s_p_%s"%(INDEX, min_eps_veri, p_of_veri))
#logging.info("i#_%s_eps_%s\n"%(INDEX, EPSILON))
logging.info("i#_%s_eps_%s_p_%s"%(INDEX, min_eps_veri, p_of_veri))
"""

