import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from datahelper import *
#import logging
from itertools import product
from arguments import argparser, logging

import keras
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, GRU
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Masking, RepeatVector, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import optimizers, layers


import sys, pickle, os
import math, json, time
import decimal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from random import shuffle
from copy import deepcopy
from sklearn import preprocessing
from emetrics import get_aupr, get_cindex, get_rm2
#import pandas as pd
from testdatahelper import *


#def get_model():
    # Create a simple model.
#    inputs = keras.Input(shape=(32,))
#    outputs = keras.layers.Dense(1)(inputs)
#    model = keras.Model(inputs, outputs)
#    model.compile(optimizer="adam", loss="mean_squared_error")
#    return model

def build_combined_categorical(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):

    XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='int32') ### Buralar flagdan gelmeliii
    XTinput = Input(shape=(FLAGS.max_seq_len,), dtype='int32')

    ### SMI_EMB_DINMS  FLAGS GELMELII
    encode_smiles = Embedding(input_dim=FLAGS.charsmiset_size+1, output_dim=128, input_length=FLAGS.max_smi_len)(XDinput)
    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)


    encode_protein = Embedding(input_dim=FLAGS.charseqset_size+1, output_dim=128, input_length=FLAGS.max_seq_len)(XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH2,  activation='relu', padding='valid',  strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)


    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein], axis=-1) #merge.Add()([encode_smiles, encode_protein])

    # Fully connected
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)


    # And add a logistic regression on top
    predictions = Dense(1, kernel_initializer='normal')(FC2) #OR no activation, rght now it's between 0-1, do I want this??? activation='sigmoid'

    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])

    interactionModel.compile(optimizer='adam', loss='mean_squared_error', metrics=[cindex_score]) #, metrics=['cindex_score']
    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/build_combined_categorical.png')

    return interactionModel

def cindex_score(y_true, y_pred):

    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g/f) #select

def prepare_interaction_pairs(XD, XT,  Y, rows, cols):
    drugs = []
    targets = []
    targetscls = []
    affinity=[] 
        
    for pair_ind in range(len(rows)):

            drug = XD[rows[pair_ind]]
            drugs.append(drug)

            target=XT[cols[pair_ind]]
            targets.append(target)

            affinity.append(Y[rows[pair_ind],cols[pair_ind]])

    drug_data = np.stack(drugs)
    target_data = np.stack(targets)

    return drug_data,target_data,  affinity

def experiment(FLAGS, perfmeasure, deepmethod, foldcount=6): #5-fold cross validation + test

    #Input
    #XD: [drugs, features] sized array (features may also be similarities with other drugs
    #XT: [targets, features] sized array (features may also be similarities with other targets
    #Y: interaction values, can be real values or binary (+1, -1), insert value float("nan") for unknown entries
    #perfmeasure: function that takes as input a list of correct and predicted outputs, and returns performance
    #higher values should be better, so if using error measures use instead e.g. the inverse -error(Y, P)
    #foldcount: number of cross-validation folds for settings 1-3, setting 4 always runs 3x3 cross-validation


    dataset = DataSet( fpath = FLAGS.train_path,
    				   fpath_test = FLAGS.test_path,
                      setting_no = FLAGS.problem_type,
                      seqlen = FLAGS.max_seq_len,
                      smilen = FLAGS.max_smi_len,
                      need_shuffle = False )
    # set character set size
    FLAGS.charseqset_size = dataset.charseqset_size
    FLAGS.charsmiset_size = dataset.charsmiset_size

    #XD, XT, Y = dataset.parse_data(fpath = FLAGS.dataset_path)
    tr_XD, tr_XT, tr_Y, te_XD, te_XT, te_Y = dataset.parse_train_test_data(FLAGS)

    tr_XD = np.asarray(tr_XD)
    tr_XT = np.asarray(tr_XT)
    tr_Y = np.asarray(tr_Y)

    te_XD = np.asarray(te_XD)
    te_XT = np.asarray(te_XT)
    te_Y = np.asarray(te_Y)

    tr_drugcount = tr_XD.shape[0]
    print("train drugs: ", tr_drugcount)
    tr_targetcount = tr_XT.shape[0]
    print("train targets: ", tr_targetcount)

    te_drugcount = te_XD.shape[0]
    print("test drugs: ", te_drugcount)
    te_targetcount = te_XT.shape[0]
    print("test targets: ", te_targetcount)

    FLAGS.drug_count = tr_drugcount
    FLAGS.target_count = tr_targetcount



    #if not os.path.exists(figdir):
    #    os.makedirs(figdir)

    print(FLAGS.log_dir)

    dependencies = {
      'cindex_score': cindex_score
    }

    #reconstructed_model = keras.models.load_model("my_model", custom_objects=dependencies)
    reconstructed_model = tf.keras.models.load_model('./saved_model', custom_objects=dependencies)

    tr_label_row_inds, tr_label_col_inds = np.where(np.isnan(tr_Y)==False)  #basically finds the point address of affinity [x,y]
    te_label_row_inds, te_label_col_inds = np.where(np.isnan(te_Y)==False)  #basically finds the point address of affinity [x,y
    test_set, outer_train_sets = dataset.read_sets(FLAGS)
    valinds=test_set
    labeledinds=outer_train_sets

    trrows = tr_label_row_inds[labeledinds]
    trcols = tr_label_col_inds[labeledinds]
    train_drugs, train_prots,  train_Y = prepare_interaction_pairs(tr_XD, tr_XT, tr_Y, trrows, trcols)

    terows = te_label_row_inds[valinds]
    tecols = te_label_col_inds[valinds]
    #print("terows", str(terows), str(len(terows)))
    #print("tecols", str(tecols), str(len(tecols)))
    
    val_drugs, val_prots,  val_Y = prepare_interaction_pairs(te_XD, te_XT,  te_Y, terows, tecols)

    pred_y = reconstructed_model.predict([np.array(train_drugs), np.array(train_prots) ])
    print(pred_y[1:10])
    print(train_Y[1:10])

    #perfmeasure = get_cindex
    #rperf = get_cindex(train_Y, pred_y)
    #print(rperf)

    #err = sum(np.absolute(np.array(train_Y) - np.array(pred_y)))/len(train_Y)
    #model = build_combined_categorical(FLAGS, FLAGS.num_windows, FLAGS.smi_window_lengths, FLAGS.seq_window_lengths)
    #print(model.predict([np.array(val_drugs), np.array(val_prots)]))

    #S1_avgperf, S1_avgloss, S1_teststd = nfold_1_2_3_setting_sample(tr_XD, tr_XT,  tr_Y, te_XD, te_XT, te_Y,
    #                                                                 perfmeasure, deepmethod, FLAGS, dataset)

    #logging("Setting " + str(FLAGS.problem_type), FLAGS)
    #logging("avg_perf = %.5f,  avg_mse = %.5f, std = %.5f" %
    #        (S1_avgperf, S1_avgloss, S1_teststd), FLAGS)

def run_regression( FLAGS ):

    perfmeasure = get_cindex
    deepmethod = build_combined_categorical

    experiment(FLAGS, perfmeasure, deepmethod)


if __name__=="__main__":

    FLAGS = argparser()
    FLAGS.log_dir = FLAGS.log_dir + str(time.time()) + "/"

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    prepare_new_data(FLAGS.test_path, test=True)
    #prepare_new_data(FLAGS.train_path, test=False) #Uncomment this if you also have a new training data

    logging(str(FLAGS), FLAGS)
    run_regression( FLAGS )


'''
FLAGS, param1value, param2value, param3value
paramset1 = FLAGS.num_windows                              #[32]#[32,  512] #[32, 128]  # filter numbers
paramset2 = FLAGS.smi_window_lengths                               #[4, 8]#[4,  32] #[4,  8] #filter length smi
paramset3 = FLAGS.seq_window_lengths                               #[8, 12]#[64,  256] #[64, 192]#[8, 192, 384

model = build_combined_categorical(FLAGS, param1value, param2value, param3value)
prepare_new_data(FLAGS.test_path, test=True)
# Train the model.
#test_input = np.random.random((128, 32))
#test_target = np.random.random((128, 1))

#model.fit(test_input, test_target)

# Calling `save('my_model')` creates a SavedModel folder `my_model`.
#model.save("my_model")

# It can be used to reconstruct the model identically.
reconstructed_model = keras.models.load_model("my_model")

# Let's check:
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)

# The reconstructed model is already compiled and has retained the optimizer
# state, so training can resume:
reconstructed_model.fit(test_input, test_target)
'''
