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
                    
def noise(s):
    """
    AWGN channel
    Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
    returns the noise vector 'n' that is added to the signal 's' and the power spectral density N0 of noise added
    Parameters:
        s : input/transmitted signal vector
        SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
        L : oversampling factor (applicable for noisy_sig simulation) default L = 1.
    Returns:
        r : received signal vector (r=s+n)
    """
    SNRdB=5
    L=1
    gamma = 10**(SNRdB/10) #SNR to linear scale
    P=L*np.sum(np.sum(abs(s)**2))/len(s) # if s is a matrix [MxN]
    N0=P/gamma # Find the noise spectral density
    # if isrealobj(s):# check if input is real/complex object type
    #     n = sqrt(N0/2)*standard_normal(s.shape) # computed noise
    # # else:
    # #     n = sqrt(N0/2)*(standard_normal(s.shape)+1j*standard_normal(s.shape))
    # r = s + n # received signal
    return s

def preprocess_noisy_outer(iq, cfo):
    """
    AWGN Noise into the signal for data augmentation
    """
    cfo_mean = 0
    noise_std = 0
    noise_mean = 0
    # if np.random.binomial(1, 1)  == 1:
    #     cfo_std = cfo
    # else:
    #     cfo_std = 0
    SNRdB= np.random.normal(scale = noise_std, loc = noise_mean)
    normalized_freq = cfo #np.random.normal(scale = cfo_std, loc = cfo_mean)
    L=1
    if int(noise_std) != 0:
        gamma = 10**(SNRdB/10) #SNR to linear scale

        P=L*tf.math.reduce_sum(tf.math.reduce_sum(abs(iq)**2))/len(iq) # if s is a matrix [MxN]
        N0=P/gamma 
        n = tf.math.sqrt(N0/2)*(tf.random.normal(iq.shape))
        noisy_sig = iq + n
        noisy_sig = tf.convert_to_tensor(noisy_sig)
    else: 
        noisy_sig = tf.convert_to_tensor(iq)
    ## Adding a CFO

    u_i_cos = noisy_sig[0]*np.cos(2*np.pi*(normalized_freq)* np.arange(128)) # I
    u_i_sin = noisy_sig[0]*np.sin(2*np.pi*(normalized_freq)* np.arange(128)) # I
    
    u_q_sin = noisy_sig[1]*np.sin(2*np.pi*(normalized_freq)* np.arange(128)) # Q
    u_q_cos = noisy_sig[1]*np.cos(2*np.pi*(normalized_freq)* np.arange(128)) # Q

    noisy_sig = tf.stack([tf.subtract(u_i_cos, u_q_sin), tf.add(u_i_sin, u_q_cos)])
    
    #nois = tf.convert_to_tensor(output_noisy_sig)
    result = rescale(tf.stack([tf_outer(noisy_sig[0], noisy_sig[0]), tf_outer(noisy_sig[1], noisy_sig[1]), tf_outer(noisy_sig[0], noisy_sig[1])], axis=2))    
    return result


