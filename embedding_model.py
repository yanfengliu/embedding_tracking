import keras.backend as K
import tensorflow as tf
from keras.layers import Concatenate, Input, Lambda, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Flatten
from keras.layers.pooling import AveragePooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2

from deeplabv3.model import Deeplabv3


def embedding_module(x, num_filter, embedding_dim, weight_decay=1E-5):
    for i in range(int(len(num_filter))):
        x = Conv2D(num_filter[i], (3, 3),
                kernel_initializer="he_uniform",
                padding="same",
                activation="relu",
                kernel_regularizer=l2(weight_decay))(x)

    x = Conv2D(embedding_dim, (3, 3),
               kernel_initializer="he_uniform",
               padding="same",
               kernel_regularizer=l2(weight_decay))(x)

    return x


def softmax_module(x, num_filter, num_class, weight_decay=1E-5):
    for i in range(int(len(num_filter))):
        x = Conv2D(num_filter[i], (3, 3),
                kernel_initializer="he_uniform",
                padding="same",
                activation="relu",
                kernel_regularizer=l2(weight_decay))(x)

    x = Conv2D(filters=num_class, 
               kernel_size=(3, 3),
               kernel_initializer="he_uniform",
               padding="same",
               activation='softmax',
               kernel_regularizer=l2(weight_decay))(x)

    return x


def dimension_conversion_module(x, out_channels, weight_decay=1E-5):
    x = Conv2D(filters=out_channels, 
               kernel_size=(1, 1),
               kernel_initializer="he_uniform",
               padding="same",
               activation='softmax',
               kernel_regularizer=l2(weight_decay))(x)
    return x

def ImageEmbeddingModel(params):
    img_size            = params.IMG_SIZE
    backbone            = params.BACKBONE
    num_filter          = params.NUM_FILTER
    num_classes         = params.NUM_CLASSES
    embedding_dim       = params.EMBEDDING_DIM
    
    deeplab_model       = Deeplabv3(input_shape = (img_size, img_size, 3), backbone = backbone)
    inputs              = deeplab_model.input
    middle              = deeplab_model.get_layer(deeplab_model.layers[-3].name).output
    classification      = softmax_module(middle, num_filter, num_classes)
    instance_embedding  = embedding_module(middle, num_filter, embedding_dim)
    final_results       = Concatenate(axis=-1)([classification,
                                                instance_embedding])
    model = Model(inputs = inputs, outputs = final_results)
    return model


def SequenceEmbeddingModel(params):
    img_size                = params.IMG_SIZE
    backbone                = params.BACKBONE
    num_filter              = params.NUM_FILTER
    num_classes             = params.NUM_CLASSES
    embedding_dim           = params.EMBEDDING_DIM
    
    # model definition
    deeplab_model = Deeplabv3(
        weights = None, 
        input_shape = (img_size, img_size, 6), 
        backbone = backbone)

    # inputs
    img_inputs = deeplab_model.input

    # intermediate representations
    middle = deeplab_model.get_layer(deeplab_model.layers[-3].name).output

    # outputs
    class_mask              = softmax_module(middle, num_filter, num_classes)
    class_mask_occ          = softmax_module(middle, num_filter, num_classes)
    class_mask_prev         = softmax_module(middle, num_filter, num_classes)
    class_mask_prev_occ     = softmax_module(middle, num_filter, num_classes)
    instance_emb            = embedding_module(middle, num_filter, embedding_dim)
    instance_emb_occ        = embedding_module(middle, num_filter, embedding_dim)
    instance_emb_prev       = embedding_module(middle, num_filter, embedding_dim)
    instance_emb_prev_occ   = embedding_module(middle, num_filter, embedding_dim)
    optical_flow            = embedding_module(middle, num_filter, 2)

    # concatenate outputs
    combined_output = Concatenate(axis=-1)([
        class_mask           ,
        class_mask_occ       ,
        class_mask_prev      ,
        class_mask_prev_occ  ,
        instance_emb         ,
        instance_emb_occ     ,
        instance_emb_prev    ,
        instance_emb_prev_occ,
        optical_flow         ])

    # build model
    model = Model(inputs = img_inputs, outputs = combined_output)
    return model

