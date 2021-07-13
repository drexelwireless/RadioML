import numpy as np
import tensorflow as tf

def rescale(x : np.ndarray) -> np.ndarray:
    """Rescale a NumPy array so its values are in the range [0,1]"""
    amin = tf.math.reduce_min(x)
    amax = tf.math.reduce_max(x)
    return (x - amin) / (amax - amin)

def tf_outer(x : np.ndarray, y : np.ndarray) -> np.ndarray:
    """Compute outer product using TensorFlow.

    Args:
        x: A vector
        y: A vector

    Returns:
        The outer product of x and y
    """
    return tf.tensordot(x, y, axes=0)

def gasf(x):
    """Compute Gramian angular summation field

    Args:
        x: A vector of size N

    Returns:
        Gramian angular summation field of shape (N, N).
    """
    y = tf.sqrt(1 - x ** 2)

    return tf_outer(x, x) - tf_outer(y, y)

def gadf(x):
    """Compute Gramian angular difference field

    Args:
        x: A vector of size N

    Returns:
        Gramian angular difference field of shape (N, N).
    """
    y = tf.sqrt(1 - x ** 2)

    return tf_outer(y, x) - tf_outer(x, y)

def preprocess_outer(iq : np.ndarray) -> np.ndarray:
    """Take IQ data of shape (2, N) and compute outer product.

    Args:
        iq: IQ data of shape (2,N)

    Returns:
        Array of shape (3,N) with three channels: the outer product of I and
        I, the outer product of Q and Q, and the outer product of I and Q.
        All channels are jointly scaled to be in the range [0,1].
    """
    result = tf.stack([ tf_outer(iq[0], iq[0])
                      , tf_outer(iq[1], iq[1])
                      , tf_outer(iq[0], iq[1])],
                      axis=2)

    return rescale(result)

def preprocess_gasf(iq : np.ndarray) -> np.ndarray:
    """Take IQ data of shape (2, N) and GASF on i channel.

    Args:
        iq: IQ data of shape (2,N)

    Returns:
        Array of shape (3,N) with three channels: the GASF of I, the outer
        product of Q and Q, and the outer product of I and Q. All channels are
        jointly scaled to be in the range [0,1].
    """
    # Computer full outer product and jointly scale it
    result = tf.stack([ tf_outer(iq[0], iq[0])
                      , tf_outer(iq[1], iq[1])
                      , tf_outer(iq[0], iq[1])],
                      axis=2)

    result = rescale(result)

    # Replace outer product of I and I with GASF
    return tf.stack([ rescale(gasf(iq[0]))
                    , result[:,:,1]
                    , result[:,:,2]],
                    axis=2)

def preprocess_gadf(iq : np.ndarray) -> np.ndarray:
    """Take IQ data of shape (2, N) and GADF on i channel.

    Args:
        iq: IQ data of shape (2,N)

    Returns:
        Array of shape (3,N) with three channels: the GADF of I, the outer
        product of Q and Q, and the outer product of I and Q. All channels are
        jointly scaled to be in the range [0,1].
    """
    # Computer full outer product and jointly scale it
    result = tf.stack([ tf_outer(iq[0], iq[0])
                      , tf_outer(iq[1], iq[1])
                      , tf_outer(iq[0], iq[1])],
                      axis=2)

    result = rescale(result)

    # Replace outer product of I and I with GASF
    return tf.stack([ rescale(gadf(iq[0]))
                    , result[:,:,1]
                    , result[:,:,2]],
                    axis=2)
