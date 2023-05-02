import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix



import dask.dataframe as dd
import csv

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.lite as tflite
import tensorflow_model_optimization as tfmot
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, SpatialDropout1D, Reshape, \
                         Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score,mean_squared_error,mean_absolute_error)

from sklearn import metrics
from sklearn.preprocessing import Normalizer

from keras import callbacks
from keras.callbacks import CSVLogger
from tensorflow.python.keras import backend as K