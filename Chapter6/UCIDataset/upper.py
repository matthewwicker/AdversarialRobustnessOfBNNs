# Author: Matthew Wicker

# Alright, hold on to your socks this one might be more rough...
import numpy as np
import sys
import os

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
from datasets import Dataset

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--imnum")
parser.add_argument("--dataset")

args = parser.parse_args()
imnum = int(args.imnum)
dataset_string = str(args.dataset)
INDEX = imnum
"""
if(dataset_string == "boston1" or dataset_string == "energy1"):
    MARGIN = 3.5
    SAMPLES = 6000

elif(dataset_string == "concrete1" or  dataset_string == "powerplant1"):
    MARGIN = 3.5
    SAMPLES = 6000

elif(dataset_string == "yacht1"):
    MARGIN = 3.5
    SAMPLES = 6000

elif(dataset_string == "wine1"):
    MARGIN = 3.5
    SAMPLES = 6000

elif(dataset_string == "kin8nm1" or dataset_string == "naval1"):
    MARGIN = 3.5
    SAMPLES = 6000
"""
#MARGIN /= 1.25
#SAMPLES *= 20
MARGIN = 2.0
SAMPLES = 1500

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
DELTA = 0.075 * output_range
print("HERE IS DELTA: ", DELTA)
# Dataset information:
in_dims = X_train.shape[1]

# Dataset information:
in_dims = X_train.shape[1]

import numpy as np
# Load in approximate posterior distribution
bayes_model = PosteriorModel("PosteriorScaled/VOGN_%s_Model"%(dataset_string))

# Lets define our correctness property (e.g. correct class)
TRUE_VALUE = y_test[INDEX]
EPSILON = 0.075


def predicate_safe(iml, imu, ol, ou):
    #error_above = float(np.squeeze(ou) - TRUE_VALUE)
    #error_below = float(TRUE_VALUE - np.squeeze(ol))
    #error_delta = abs(error_above) + abs(error_below)
    #if(error_delta > (2*DELTA) ):
    bound_above = TRUE_VALUE + DELTA
    bound_below = TRUE_VALUE - DELTA
    if(ou <= bound_below or ol >= bound_above):
        return True
    else:
        return False

# We should do clipping, but its 4am now and I am lazy
img = np.asarray([X_test[INDEX]])
img_upper = np.asarray([X_test[INDEX]+(input_range*EPSILON)])
img_lower = np.asarray([X_test[INDEX]-(input_range*EPSILON)])


import json
save_dir = "ReduxLogs"
# We start with epsilon = 0.0 and increase it as we go.
p_lower, _ = IBP_prob(bayes_model, img_upper, img_upper, MARGIN, SAMPLES, predicate=predicate_safe, inflate=1.0)
#p_lower2, _ = IBP_prob(bayes_model, img_lower, img_lower, MARGIN, SAMPLES, predicate=predicate_safe, inflate=2.5)
#p_lower = max(p_lower1, p_lower2)
p_lower = 1-p_lower
print("Initial Safety Probability: ", p_lower)
record = {"Index":INDEX, "Upper":p_lower, "Samples":SAMPLES, "Margin":MARGIN, "Epsilon":EPSILON, "Delta":DELTA}
with open("%s/%s_upper.log"%(save_dir, dataset_string), 'a') as f:
    json.dump(record, f)
    f.write(os.linesep)

"""
if(p_lower < 0.9):
    import logging
    logging.basicConfig(filename="MSELogs/%s_eps.log"%(dataset_string),level=logging.DEBUG)
    #print("i#_%s_eps_%s_p_%s"%(INDEX, max_eps_veri, p_of_veri))
    logging.info("i#_%s_eps_%s_p_%s"%(INDEX, 0.0, 0.0))
    sys.exit()

stepsize = 0.10
max_eps_veri = 0.0
p_of_veri = 0.0

for i in range(4):
    if(p_lower >= 0.9):
        max_eps_veri = max(EPSILON, max_eps_veri)
        p_of_veri = p_lower #max(EPSILON, max_eps_veri)
        EPSILON += stepsize
        stepsize /= 2
        img_upper = np.asarray([X_test[INDEX]+(input_range*EPSILON)])
        img_lower = np.asarray([X_test[INDEX]-(input_range*EPSILON)])
    else:
        EPSILON -= stepsize
        stepsize /= 2
        img_upper = np.asarray([X_test[INDEX]+(input_range*EPSILON)])
        img_lower = np.asarray([X_test[INDEX]-(input_range*EPSILON)])
    print("Computing with epsilon: ", EPSILON)
    print("Stepsize: ", stepsize)
    p_lower, _ = IBP_prob(bayes_model, img_lower, img_upper, MARGIN, SAMPLES, predicate=predicate_safe, inflate=2.5)
    print("Probability: ", p_lower)

import logging
logging.basicConfig(filename="MSELogs/%s_eps.log"%(dataset_string),level=logging.DEBUG)
print("i#_%s_eps_%s_p_%s"%(INDEX, max_eps_veri, p_of_veri))
#logging.info("i#_%s_eps_%s\n"%(INDEX, EPSILON))
logging.info("i#_%s_eps_%s_p_%s"%(INDEX, max_eps_veri, p_of_veri))



"""
