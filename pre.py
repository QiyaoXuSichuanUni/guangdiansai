# common
import os
import keras
import keras.utils
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
from tensorflow import image as tfi

# Data
from keras.utils import load_img, img_to_array
from keras.utils import to_categorical

# Data Viz
import matplotlib.pyplot as plt

# Model 
from keras.models import Model
from keras.layers import Layer
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import UpSampling2D
from keras.layers import concatenate
from keras.layers import Add
from keras.layers import Multiply
from keras.layers import Input
from keras.layers import MaxPool2D
from keras.layers import BatchNormalization

# Callbacks 
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

# Metrics
from keras.metrics import MeanIoU
from keras.models import load_model
from attention_unet import EncoderBlock, DecoderBlock, AttentionGate

# 加载模型时注册自定义层
custom_objects = {
    'EncoderBlock': EncoderBlock,
    'DecoderBlock': DecoderBlock,
    'AttentionGate': AttentionGate
}

model = load_model('AttentionCustomUNet.h5', custom_objects=custom_objects)
input_img = load_img('path_to_your_image',target_size=(256,256))
img = img_to_array(img_to_array).astype('float32')/255


pre_array = model.predict(img)
img_out = keras.utils.array_to_img(pre_array)
img_out.save('path_to your_img')
