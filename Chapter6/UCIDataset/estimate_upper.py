# Author: Matthew Wicker
print("HERE WE GO")
# Alright, hold on to your socks this one might be more rough...
import numpy as np
import sys


sys.path.append('../../')
import deepbayesHF
import deepbayesHF.optimizers as optimizers

# All of the inputs are basically the same except for this
from deepbayesHF import PosteriorModel
from deepbayesHF.analyzers import IBP_prob
from deepbayesHF.analyzers import IBP_upper
from deepbayesHF.analyzers import FGSM
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

print("Done with imports")
from datasets import Dataset


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--imnum")
parser.add_argument("--dataset")

args = parser.parse_args()
imnum = int(args.imnum)
dataset_string = str(args.dataset)
INDEX = imnum



if(dataset_string == "boston1" or dataset_string == "energy1"):
    MARGIN = 2.0
    SAMPLES = 1000

elif(dataset_string == "concrete1" or  dataset_string == "powerplant1"):
    MARGIN = 2.0
    SAMPLES = 1000

elif(dataset_string == "yacht1"):
    MARGIN = 2.0
    SAMPLES = 1000

elif(dataset_string == "wine1"):
    MARGIN = 1.5
    SAMPLES = 2000

elif(dataset_string == "kin8nm1" or dataset_string == "naval1"):
    MARGIN = 1.5
    SAMPLES = 2000



data = Dataset(dataset_string)
from sklearn.preprocessing import StandardScaler

X_train = np.asarray(data.train_set.train_data)
y_train = np.asarray(data.train_set.train_labels.reshape(-1,1))

X_test = np.asarray(data.test_set.test_data)
y_test = np.asarray(data.test_set.test_labels.reshape(-1,1))

X_scaler = StandardScaler()
X_scaler.fit(X_train)
X_train, X_test = X_scaler.transform(X_train), X_scaler.transform(X_test)
print("Sucessfully loaded dataset %s \t \t with train shapes %s, %s"%(dataset_string, X_train.shape, y_train.shape))

input_range = np.max(X_test, axis=0) - np.min(X_test, axis=0)
input_range /= 2
print("Input range: ", input_range)
output_range = np.max(y_test) - np.min(y_test)
print("Output range: ", np.max(y_test) - np.min(y_test))
DELTA = 0.05 * output_range
print("HERE IS DELTA: ", DELTA)
# Dataset information:
in_dims = X_train.shape[1]


input_range = np.max(X_test, axis=0) - np.min(X_test, axis=0)
#input_range /= 2
print("Input range: ", input_range)
output_range = np.max(y_test) - np.min(y_test)
print("Output range: ", np.max(y_test) - np.min(y_test))
DELTA = 0.1 * output_range

# Dataset information:
in_dims = X_train.shape[1]

import numpy as np
# Load in approximate posterior distribution
bayes_model = PosteriorModel("Posteriors/VOGN_%s_Model"%(dataset_string))
"""
 the parameter 'n' below is the number of samples from the posterior
 used to approximate the posterior predictive distribution
"""
#y_pred = bayes_model.predict(X_test, n=35)
#check_rmse = tf.keras.metrics.RootMeanSquaredError(name="eval_rmse")
#check_rmse(y_test, y_pred)
#print("Loaded Model RMSE: ", check_rmse.result())


# Lets define our correctness property (e.g. correct class)
TRUE_VALUE = y_test[INDEX]

def predicate_safe(iml, imu, ol, ou):
    error_above = float(np.squeeze(ou) - TRUE_VALUE)
    error_below = float(TRUE_VALUE - np.squeeze(ol))
    error_delta = abs(error_above) + abs(error_below)
    if(error_delta > (2*DELTA) ):
        return False
    else:
        return True

def predicate_unsafe(iml, imu, ol, ou):
    bound_above = TRUE_VALUE + DELTA
    bound_below = TRUE_VALUE - DELTA
    if(bound_above <= ol or bound_below >= ou):
        return True
    else:
        return False

# We should do clipping, but its 4am now and I am lazy
stepsize = 0.25
EPSILON = 0.5
min_eps_veri = 1.0
p_of_veri = 1.0

# Generate a series of attacks
from deepbayesHF import analyzers
meth = analyzers.PGD
loss = tf.keras.losses.MeanSquaredError()

attacks = []
img = np.asarray([X_test[INDEX]])
img_upper = np.asarray([X_test[INDEX]+(input_range*EPSILON)])
img_lower = np.asarray([X_test[INDEX]-(input_range*EPSILON)])
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

for i in range(4):
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
    p_upper, _ = IBP_upper(bayes_model, img, img, MARGIN, SAMPLES, loss_fn=loss, eps=EPSILON, predicate=predicate_unsafe, inputs=attacks, inflate=2.5, mod_option=10)
    p_upper = 1 - p_upper
    print("Probability: ", p_upper)

import logging
logging.basicConfig(filename="MURLogs/%s_upper_eps.log"%(dataset_string),level=logging.DEBUG)
print("i#_%s_eps_%s_p_%s"%(INDEX, min_eps_veri, p_of_veri))
#logging.info("i#_%s_eps_%s\n"%(INDEX, EPSILON))
logging.info("i#_%s_eps_%s_p_%s"%(INDEX, min_eps_veri, p_of_veri))


