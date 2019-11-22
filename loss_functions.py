import keras.backend as K
import tensorflow as tf


def sequence_loss_with_params(params):
    def double_frame_sequence_loss(y_true, y_pred):
        # hyperparameters
        delta_var     = params.DELTA_VAR
        delta_d       = params.DELTA_D
        class_num     = params.NUM_CLASSES
        embedding_dim = params.EMBEDDING_DIM

        # unpack ground truth contents
        class_mask_gt      = y_true[:, :, :, 0]
        prev_class_mask_gt = y_true[:, :, :, 1]
        identity_mask      = y_true[:, :, :, 2]
        prev_identity_mask = y_true[:, :, :, 3]
        optical_flow_gt    = y_true[:, :, :, 4:6]

        # y_pred
        class_mask_pred      = y_pred[:, :, :, :class_num]
        prev_class_mask_pred = y_pred[:, :, :, class_num:(class_num * 2)]
        instance_emb         = y_pred[:, :, :, (class_num * 2):(class_num * 2 + embedding_dim)]
        prev_instance_emb    = y_pred[:, :, :, (class_num * 2 + embedding_dim):(class_num * 2 + embedding_dim * 2)]
        optical_flow_pred    = y_pred[:, :, :, (class_num * 2 + embedding_dim * 2):]

        # flatten and combine
        # --identity mask gt
        identity_mask_flat            = K.flatten(identity_mask)
        prev_identity_mask_flat       = K.flatten(prev_identity_mask)
        combined_identity_mask_flat   = tf.concat((identity_mask_flat, prev_identity_mask_flat), axis=0)
        # --instance embedding pred
        instance_emb_flat             = tf.reshape(instance_emb, shape=(-1, embedding_dim))
        prev_instance_emb_flat        = tf.reshape(prev_instance_emb, shape=(-1, embedding_dim))
        combined_instance_emb_flat    = tf.concat((instance_emb_flat, prev_instance_emb_flat), axis=0)
        # --class mask gt
        class_mask_gt_flat            = K.flatten(class_mask_gt)
        prev_class_mask_gt_flat       = K.flatten(prev_class_mask_gt)
        combined_class_mask_gt_flat   = tf.concat((class_mask_gt_flat, prev_class_mask_gt_flat), axis=0)
        # --class mask pred
        class_mask_pred_flat          = tf.reshape(class_mask_pred, shape=(-1, class_num))
        prev_class_mask_pred_flat     = tf.reshape(prev_class_mask_pred, shape=(-1, class_num))
        combined_class_mask_pred_flat = tf.concat((class_mask_pred_flat, prev_class_mask_pred_flat), axis=0)

        # get number of pixels and clusters (without background)
        num_cluster = tf.reduce_max(combined_identity_mask_flat)
        num_cluster = tf.cast(num_cluster, tf.int32)

        # cast masks into tf.int32 for one-hot encoding
        combined_identity_mask_flat = tf.cast(combined_identity_mask_flat, tf.int32)
        combined_class_mask_gt_flat = tf.cast(combined_class_mask_gt_flat, tf.int32)

        # one-hot encoding
        combined_identity_mask_flat -= 1
        combined_identity_mask_flat_one_hot = tf.one_hot(combined_identity_mask_flat, num_cluster)
        conbined_class_mask_gt_flat_one_hot = tf.one_hot(combined_class_mask_gt_flat, class_num)

        # ignore background pixels
        non_background_idx                  = tf.greater(combined_identity_mask_flat, -1)
        combined_instance_emb_flat          = tf.boolean_mask(combined_instance_emb_flat, non_background_idx)
        combined_identity_mask_flat         = tf.boolean_mask(combined_identity_mask_flat, non_background_idx)
        combined_identity_mask_flat_one_hot = tf.boolean_mask(combined_identity_mask_flat_one_hot, non_background_idx)
        combined_class_mask_gt_flat         = tf.boolean_mask(combined_class_mask_gt_flat, non_background_idx)

        # center count
        combined_identity_mask_flat_one_hot = tf.cast(combined_identity_mask_flat_one_hot, tf.float32)
        center_count = tf.reduce_sum(combined_identity_mask_flat_one_hot, axis=0)
        # add a small number to avoid division by zero

        # variance term
        embedding_sum_by_instance = tf.matmul(
            tf.transpose(combined_instance_emb_flat), combined_identity_mask_flat_one_hot)
        centers = tf.math.divide_no_nan(embedding_sum_by_instance, center_count)
        gathered_center = tf.gather(centers, combined_identity_mask_flat, axis=1)
        gathered_center_count = tf.gather(center_count, combined_identity_mask_flat)
        combined_emb_t = tf.transpose(combined_instance_emb_flat)
        var_dist = tf.norm(combined_emb_t - gathered_center, ord=1, axis=0) - delta_var
        # changed from soft hinge loss to hard cutoff
        var_dist_pos = tf.square(tf.maximum(var_dist, 0))
        var_dist_by_instance = tf.math.divide_no_nan(var_dist_pos, gathered_center_count)
        num_cluster = tf.cast(num_cluster, tf.float32)
        variance_term = tf.math.divide_no_nan(
            tf.reduce_sum(var_dist_by_instance),
            tf.cast(num_cluster, tf.float32))

        # get instance to class mapping
        class_mask_gt = tf.expand_dims(class_mask_gt, axis=-1)
        # multiply classification with one hot flat identity mask
        combined_class_mask_gt_flat = tf.cast(combined_class_mask_gt_flat, tf.float32)
        combined_class_mask_gt_flat = tf.expand_dims(combined_class_mask_gt_flat, 1)
        filtered_class = tf.multiply(combined_identity_mask_flat_one_hot, combined_class_mask_gt_flat)
        # shrink to a 1 by num_of_cluster vector to map instance to class;
        # by reduce_max, any class other than 0 (background) stands out
        instance_to_class = tf.reduce_max(filtered_class, axis = [0])

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
            total_cluster_pair = num_cluster_by_class * (num_cluster_by_class - 1) + 1
            total_cluster_pair = tf.cast(total_cluster_pair, tf.float32)
            distance_term = tf.math.divide_no_nan(tf.reduce_sum(distance_term), total_cluster_pair)
            return distance_term


        def distance_false_fn():
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
        instance_emb_sequence_loss = variance_term + distance_term_total + 0.01 * regularization_term
        semseg_loss = K.mean(K.categorical_crossentropy(
            tf.cast(conbined_class_mask_gt_flat_one_hot, tf.float32), 
            tf.cast(combined_class_mask_pred_flat, tf.float32)))
        # masked mse for optical loss
        flow_mask = tf.greater(prev_class_mask_gt, 0)
        flow_mask = tf.cast(flow_mask, tf.float32)
        flow_mask = tf.expand_dims(flow_mask, axis = -1)
        masked_optical_flow_pred = tf.math.multiply(optical_flow_pred, flow_mask)
        optical_flow_loss = tf.reduce_mean(tf.square(masked_optical_flow_pred - optical_flow_gt))
        loss = instance_emb_sequence_loss + semseg_loss + 5 * optical_flow_loss
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
            centers = tf.math.divide_no_nan(embedding_sum_by_instance, center_count)
            gathered_center = tf.gather(centers, instance_mask_flat, axis=1)
            gathered_center_count = tf.gather(center_count, instance_mask_flat)
            combined_emb_t = tf.transpose(instance_emb_flat)
            var_dist = tf.norm(combined_emb_t - gathered_center, ord=1, axis=0) - delta_var
            # changed from soft hinge loss to hard cutoff
            var_dist_pos = tf.square(tf.maximum(var_dist, 0))
            var_dist_by_instance = tf.math.divide_no_nan(var_dist_pos, gathered_center_count)
            variance_term = tf.math.divide_no_nan(
                tf.reduce_sum(var_dist_by_instance),
                tf.cast(num_cluster, tf.float32))

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
                total_cluster_pair = num_cluster_by_class * (num_cluster_by_class - 1) + 1
                total_cluster_pair = tf.cast(total_cluster_pair, tf.float32)
                distance_term = tf.math.divide_no_nan(tf.reduce_sum(distance_term), total_cluster_pair)
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