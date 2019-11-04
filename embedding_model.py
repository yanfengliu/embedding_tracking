import keras.backend as K
import tensorflow as tf
from keras.layers import Concatenate, Input, Lambda, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Flatten
from keras.layers.pooling import AveragePooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2

from deeplabv3.model import Deeplabv3


def sequence_loss_with_params(params):
    def double_frame_sequence_loss(y_true, y_pred):
        # hyperparameters
        delta_var     = params.DELTA_VAR
        delta_d       = params.DELTA_D
        class_num     = params.NUM_CLASSES
        embedding_dim = params.EMBEDDING_DIM

        # unpack ground truth contents
        class_mask         = y_true[:, :, :, 0]
        prev_class_mask    = y_true[:, :, :, 1]
        identity_mask      = y_true[:, :, :, 2]
        prev_identity_mask = y_true[:, :, :, 3]
        optical_flow       = y_true[:, :, :, 4:5]

        # y_pred
        class_pred        = y_pred[:, :, :, :class_num]
        prev_class_pred   = y_pred[:, :, :, class_num:(class_num * 2)]
        instance_emb      = y_pred[:, :, :, (class_num * 2):(class_num * 2 + embedding_dim)]
        prev_instance_emb = y_pred[:, :, :, (class_num * 2 + embedding_dim):(class_num * 2 + embedding_dim * 2)]
        optical_flow      = y_pred[:, :, :, (class_num * 2 + embedding_dim * 2):]

        # get number of pixels and clusters (without background)
        num_cluster = tf.reduce_max(instance_mask)
        num_cluster = tf.cast(num_cluster, tf.int32)

        # one-hot encoding for mask
        instance_mask = tf.cast(instance_mask, tf.int32)
        instance_mask = instance_mask - 1
        instance_mask_one_hot = tf.one_hot(instance_mask, num_cluster)

        class_mask = tf.cast(class_mask, tf.int32)
        class_mask_one_hot = tf.one_hot(class_mask, class_num)

        # flatten
        instance_emb_flat           = tf.reshape(instance_emb, shape=(-1, embedding_dim))
        instance_mask_one_hot_flat  = tf.reshape(instance_mask_one_hot, shape=(-1, num_cluster))
        instance_mask_flat          = K.flatten(instance_mask)

        class_mask_flat             = tf.reshape(class_mask_one_hot, shape=(-1, class_num))
        class_pred_flat             = tf.reshape(class_pred, shape=(-1, class_num))

        # ignore background pixels
        non_background_idx          = tf.greater(instance_mask_flat, -1)
        instance_emb_flat           = tf.boolean_mask(instance_emb_flat, non_background_idx)
        instance_mask_flat          = tf.boolean_mask(instance_mask_flat, non_background_idx)
        instance_mask_one_hot_flat  = tf.boolean_mask(instance_mask_one_hot_flat, non_background_idx)

        # center count
        center_count = tf.reduce_sum(tf.cast(instance_mask_one_hot_flat, dtype=tf.float32), axis=0)

        # variance term
        embedding_sum_by_instance = tf.matmul(
            tf.transpose(instance_emb_flat), tf.cast(instance_mask_one_hot_flat, dtype=tf.float32))
        centers = tf.divide(embedding_sum_by_instance, center_count)
        gathered_center = tf.gather(centers, instance_mask_flat, axis=1)
        gathered_center_count = tf.gather(center_count, instance_mask_flat)
        combined_emb_t = tf.transpose(instance_emb_flat)
        var_dist = tf.norm(combined_emb_t - gathered_center, ord=1, axis=0) - delta_var
        # changed from soft hinge loss to hard cutoff
        var_dist_pos = tf.square(tf.maximum(var_dist, 0))
        var_dist_by_instance = tf.divide(var_dist_pos, gathered_center_count)
        variance_term = tf.reduce_sum(var_dist_by_instance) / tf.cast(num_cluster, tf.float32)

        # get instance to class mapping
        class_mask = tf.expand_dims(class_mask, axis=-1)
        filtered_class = tf.multiply(tf.cast(instance_mask_one_hot, tf.float32), tf.cast(class_mask, tf.float32))
        instance_to_class = tf.reduce_max(filtered_class, axis = [0, 1, 2])


        def distance_true_fn(num_cluster_by_class, centers_by_class):
            centers_row_buffer = tf.ones((embedding_dim, num_cluster_by_class, num_cluster_by_class))
            centers_by_class = tf.expand_dims(centers_by_class, axis=2)
            centers_row = tf.multiply(centers_row_buffer, centers_by_class)
            centers_col = tf.transpose(centers_row, perm=[0, 2, 1])
            dist_matrix = centers_row - centers_col
            idx2 = tf.ones((num_cluster_by_class, num_cluster_by_class))
            diag = tf.ones((1, num_cluster_by_class))
            diag = tf.reshape(diag, [-1])
            idx2 = idx2 - tf.diag(diag)
            idx2 = tf.cast(idx2, tf.bool)
            idx2 = K.flatten(idx2)
            dist_matrix = tf.reshape(dist_matrix, [embedding_dim, -1])
            dist_matrix = tf.transpose(dist_matrix)
            sampled_dist = tf.boolean_mask(dist_matrix, idx2)
            distance_term = tf.square(tf.maximum(
                2 * delta_d - tf.norm(sampled_dist, ord=1, axis=1), 0))
            distance_term = tf.reduce_sum(
                distance_term) / tf.cast(num_cluster_by_class * (num_cluster_by_class - 1) + 1, tf.float32)
            return distance_term


        def distance_false_fn():
            return 0.0

        
        def sequence_true_fn(prev_centers, centers):
            return tf.reduce_sum(tf.abs(prev_centers - centers))
        
        
        def sequence_false_fn():
            return 0.0


        distance_term_total = 0.0
        # center distance term
        for i in range(class_num-1):
            class_idx = tf.equal(instance_to_class, i+1)
            centers_transpose = tf.transpose(centers)
            centers_by_class_transpose = tf.boolean_mask(centers_transpose, class_idx)
            centers_by_class = tf.transpose(centers_by_class_transpose)
            num_cluster_by_class = tf.reduce_sum(tf.cast(class_idx, tf.float32))
            num_cluster_by_class = tf.cast(num_cluster_by_class, tf.int32)
            distance_term_subtotal = tf.cond(num_cluster_by_class > 0, 
                                            lambda: distance_true_fn(num_cluster_by_class, centers_by_class), 
                                            lambda: distance_false_fn())
            distance_term_total += distance_term_subtotal
            
        # regularization term
        regularization_term = tf.reduce_mean(tf.norm(tf.squeeze(centers), ord=1, axis=0))

        # sum up terms
        instance_emb_loss = variance_term + distance_term_total + 0.01 * regularization_term
        semseg_loss = K.mean(K.categorical_crossentropy(
            tf.cast(class_mask_flat, tf.float32), tf.cast(class_pred_flat, tf.float32)))
        instance_emb_sequence_loss = tf.cond(
            tf.constant(tf.reduce_sum(prev_centers) != 0, dtype = tf.bool),
            lambda: sequence_true_fn(prev_centers, centers),
            lambda: sequence_false_fn()
        )
        # loss = instance_emb_loss + semseg_loss + instance_emb_sequence_loss
        loss = instance_emb_loss + semseg_loss
        loss = tf.reshape(loss, [-1])

        return loss
    return double_frame_sequence_loss


def single_frame_loss_with_params(params):
    def multi_class_instance_embedding_loss(y_true, y_pred):
        # hyperparameters
        batch_size    = params.BATCH_SIZE
        delta_var     = params.DELTA_VAR
        delta_d       = params.DELTA_D
        class_num     = params.NUM_CLASSES
        embedding_dim = params.EMBEDDING_DIM

        total_loss = 0
        # unpack ground truth contents
        for j in range(batch_size):
            class_mask    = y_true[j:j+1, :, :, 0]
            instance_mask = y_true[j:j+1, :, :, 1]

            # y_pred
            class_pred   = y_pred[j:j+1, :, :, :class_num]
            instance_emb = y_pred[j:j+1, :, :, (class_num):(class_num + embedding_dim)]

            # get number of pixels and clusters (without background)
            num_cluster = tf.reduce_max(instance_mask)
            num_cluster = tf.cast(num_cluster, tf.int32)

            # one-hot encoding for mask
            instance_mask = tf.cast(instance_mask, tf.int32)
            instance_mask = instance_mask - 1
            instance_mask_one_hot = tf.one_hot(instance_mask, num_cluster)

            class_mask = tf.cast(class_mask, tf.int32)
            class_mask_one_hot = tf.one_hot(class_mask, class_num)

            # flatten
            instance_emb_flat           = tf.reshape(instance_emb, shape=(-1, embedding_dim))
            instance_mask_one_hot_flat  = tf.reshape(instance_mask_one_hot, shape=(-1, num_cluster))
            instance_mask_flat          = K.flatten(instance_mask)

            class_mask_flat             = tf.reshape(class_mask_one_hot, shape=(-1, class_num))
            class_pred_flat             = tf.reshape(class_pred, shape=(-1, class_num))

            # ignore background pixels
            non_background_idx          = tf.greater(instance_mask_flat, -1)
            instance_emb_flat           = tf.boolean_mask(instance_emb_flat, non_background_idx)
            instance_mask_flat          = tf.boolean_mask(instance_mask_flat, non_background_idx)
            instance_mask_one_hot_flat  = tf.boolean_mask(instance_mask_one_hot_flat, non_background_idx)

            # center count
            center_count = tf.reduce_sum(tf.cast(instance_mask_one_hot_flat, dtype=tf.float32), axis=0)

            # variance term
            embedding_sum_by_instance = tf.matmul(
                tf.transpose(instance_emb_flat), tf.cast(instance_mask_one_hot_flat, dtype=tf.float32))
            centers = tf.divide(embedding_sum_by_instance, center_count)
            gathered_center = tf.gather(centers, instance_mask_flat, axis=1)
            gathered_center_count = tf.gather(center_count, instance_mask_flat)
            combined_emb_t = tf.transpose(instance_emb_flat)
            var_dist = tf.norm(combined_emb_t - gathered_center, ord=1, axis=0) - delta_var
            # changed from soft hinge loss to hard cutoff
            var_dist_pos = tf.square(tf.maximum(var_dist, 0))
            var_dist_by_instance = tf.divide(var_dist_pos, gathered_center_count)
            variance_term = tf.reduce_sum(var_dist_by_instance) / tf.cast(num_cluster, tf.float32)

            # get instance to class mapping
            class_mask = tf.expand_dims(class_mask, axis=-1)
            filtered_class = tf.multiply(tf.cast(instance_mask_one_hot, tf.float32), tf.cast(class_mask, tf.float32))
            instance_to_class = tf.reduce_max(filtered_class, axis = [0, 1, 2])


            def true_fn(num_cluster_by_class, centers_by_class):
                centers_row_buffer = tf.ones((embedding_dim, num_cluster_by_class, num_cluster_by_class))
                centers_by_class = tf.expand_dims(centers_by_class, axis=2)
                centers_row = tf.multiply(centers_row_buffer, centers_by_class)
                centers_col = tf.transpose(centers_row, perm=[0, 2, 1])
                dist_matrix = centers_row - centers_col
                idx2 = tf.ones((num_cluster_by_class, num_cluster_by_class))
                diag = tf.ones((1, num_cluster_by_class))
                diag = tf.reshape(diag, [-1])
                idx2 = idx2 - tf.diag(diag)
                idx2 = tf.cast(idx2, tf.bool)
                idx2 = K.flatten(idx2)
                dist_matrix = tf.reshape(dist_matrix, [embedding_dim, -1])
                dist_matrix = tf.transpose(dist_matrix)
                sampled_dist = tf.boolean_mask(dist_matrix, idx2)
                distance_term = tf.square(tf.maximum(
                    2 * delta_d - tf.norm(sampled_dist, ord=1, axis=1), 0))
                distance_term = tf.reduce_sum(
                    distance_term) / tf.cast(num_cluster_by_class * (num_cluster_by_class - 1) + 1, tf.float32)
                return distance_term


            def false_fn():
                return 0.0


            distance_term_total = 0.0
            # center distance term
            for i in range(class_num-1):
                class_idx = tf.equal(instance_to_class, i+1)
                centers_transpose = tf.transpose(centers)
                centers_by_class_transpose = tf.boolean_mask(centers_transpose, class_idx)
                centers_by_class = tf.transpose(centers_by_class_transpose)
                num_cluster_by_class = tf.reduce_sum(tf.cast(class_idx, tf.float32))
                num_cluster_by_class = tf.cast(num_cluster_by_class, tf.int32)
                distance_term_subtotal = tf.cond(num_cluster_by_class > 0, 
                                                lambda: true_fn(num_cluster_by_class, centers_by_class), 
                                                lambda: false_fn())
                distance_term_total += distance_term_subtotal
                
            # regularization term
            regularization_term = tf.reduce_mean(tf.norm(tf.squeeze(centers), ord=1, axis=0))

            # sum up terms
            instance_emb_loss = variance_term + distance_term_total + 0.01 * regularization_term
            semseg_loss = K.mean(K.categorical_crossentropy(
                tf.cast(class_mask_flat, tf.float32), tf.cast(class_pred_flat, tf.float32)))
            loss = instance_emb_loss + semseg_loss
            loss = tf.reshape(loss, [-1])
            total_loss += loss
            
        total_loss = total_loss / batch_size

        return total_loss
    return multi_class_instance_embedding_loss


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
    deeplab_model           = Deeplabv3(input_shape = (img_size, img_size, 6 + embedding_dim), backbone = backbone)

    # inputs
    img_inputs              = deeplab_model.input

    # intermediate representations
    middle                  = deeplab_model.get_layer(deeplab_model.layers[-3].name).output

    # outputs
    instance_embedding      = embedding_module(middle, num_filter, embedding_dim)
    semantic_segmentation   = softmax_module(  middle, num_filter, num_classes)
    optical_flow            = embedding_module(middle, num_filter, 2)

    # concatenate outputs
    combined_output         = Concatenate(axis=-1)([
        instance_embedding,
        semantic_segmentation,
        optical_flow])

    # build model
    model = Model(inputs = [img_inputs], outputs = combined_output)
    return model

