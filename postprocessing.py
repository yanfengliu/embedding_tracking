import numpy as np
import time


def ETH_mean_shift(data, mask, threshold=0.5):
    """
    Perform adapted fast mean shift on pixel embedding output. Based on
    https://arxiv.org/abs/1708.02551

    Inputs:
    =======
    data: np.array -- size: [width, length, embedding_dimension];
        pixel mebedding output of the network
    mask: np.array -- size: [width, length, 1]; mask with background
        as = 0, and foreground as greater than 0
    threshold: float -- distance threshold to decide if two embeddings
        belong to the same instance

    Outputs:
    ========
    clustered: np.array -- clustering result with consecutive non-negative
        integers representing distinct instances
    """

    MS_threshold = 0.0

    x = np.squeeze(data)
    if len(x.shape) == 2:
        x = np.expand_dims(x, -1)
    x_shape = np.array(x.shape)

    # get dimensions
    embedding_dim = x_shape[2]
    num_pixels = x_shape[0] * x_shape[1]

    # flatten data
    x = np.reshape(x, newshape=(-1, embedding_dim))
    x_shape_flat = np.array(x.shape)
    foreground_mask = mask > 0
    mask_flat = np.reshape(foreground_mask, [-1])

    # record where the foreground pixels are taken from,
    # to reshape them back later
    full_idx = np.array(list(range(num_pixels)))
    foreground_idx = full_idx[mask_flat]
    foreground_x = np.take(x, foreground_idx, axis=0)
    N = foreground_x.shape[0]
    idx = np.array(list(range(N)))
    idx_pool = np.ones(shape=(N,), dtype=bool)
    label = 1
    label_array = np.zeros(shape=(1, N))

    # iterative step
    old_time = time.time()
    timeout = 5
    while time.time() - old_time < timeout and np.sum(idx_pool) > 0:
        # randomly select an unlabeled embedding
        available_idx = idx[idx_pool]
        next_idx = available_idx[np.random.randint(0, high=len(available_idx))]
        embedding = foreground_x[next_idx, :]
        dist_array = np.linalg.norm(foreground_x - embedding, ord=1, axis=1)
        within_idx = dist_array < threshold
        within_embedding = foreground_x[within_idx, :]
        new_cluster_mean = np.mean(within_embedding, axis=0)
        old_cluster_mean = new_cluster_mean + threshold + 1
        idx_pool[next_idx] = False

        # threshold around the cluster mean until convergence
        step_dist = np.linalg.norm(new_cluster_mean - old_cluster_mean, ord=1)
        while time.time() - old_time < timeout and step_dist > MS_threshold:
            dist_array = np.linalg.norm(
                foreground_x - new_cluster_mean, ord=1, axis=1)
            within_idx = dist_array < threshold
            within_embedding = foreground_x[within_idx, :]
            old_cluster_mean = new_cluster_mean
            new_cluster_mean = np.mean(within_embedding, axis=0)
            step_dist = np.linalg.norm(
                new_cluster_mean - old_cluster_mean, ord=1)

        # threshold around the converged mean and label all embeddings within
        dist_array = np.linalg.norm(
            foreground_x - new_cluster_mean, ord=1, axis=1)
        within_idx = dist_array < threshold
        foreground_x[within_idx, :] = new_cluster_mean
        idx_pool[within_idx] = False
        within_idx = np.expand_dims(within_idx, 0)
        label_array[within_idx] = label
        label_array[0, next_idx] = label
        label = label + 1

    # reshape and display
    x_shape_flat[-1] = 1
    label_array = np.squeeze(label_array)
    label_array = np.expand_dims(label_array, 1)
    full_label = np.zeros((x_shape_flat))
    full_label[foreground_idx] = label_array
    x_shape[-1] = 1
    full_label = np.reshape(full_label, x_shape[:-1])

    # change to consecutive integers starting at zero
    count = 0
    clustered = np.copy(full_label)
    for value in np.unique(full_label):
        clustered[full_label == value] = count
        count = count + 1

    return clustered