"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    half_Hk, half_Wk = Hk // 2, Wk // 2
    image_padded = zero_pad(image, half_Hk, half_Wk)
    for i in range(Hi):
    	for j in range(Wi):
    		for k in range(Hk):
    			for l in range(Wk):
    				out[i, j] += image_padded[i-k+Hk-1, j-l+Wk-1] * kernel[k, l]
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out_H, out_W = H + 2 * pad_height, W + 2 * pad_width
    out = np.zeros((out_H, out_W), dtype=image.dtype)
    out[pad_height:pad_height + H, pad_width:pad_width + W] = image
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    half_Hk, half_Wk = Hk // 2, Wk // 2
    image_padded = zero_pad(image, half_Hk, half_Wk)
    kernel_flipped = np.flipud(np.fliplr(kernel))
    for i in range(Hi):
    	for j in range(Wi):
    		out[i, j] = np.sum(image_padded[i:i + Hk, j:j + Wk] * kernel_flipped)
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    kernel_flipped = np.flipud(np.fliplr(kernel))
    half_Hk, half_Wk = Hk // 2, Wk // 2
    image_padded = zero_pad(image, half_Hk, half_Wk)
    for i in range(Hi):
    	for j in range(Wi):
    		out[i, j] = np.ravel(image_padded[i:i + Hk, j:j + Wk]) @ np.ravel(kernel_flipped)
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g_flipped = np.flipud(np.fliplr(g))
    out = conv_faster(f, g)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g_normalized = g - np.mean(g)
    out = cross_correlation(f, g_normalized)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g = g[:-1, :] # make the dimension of g is odd number
    g_normalized = (g - np.mean(g)) / np.std(g)
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    half_Hg, half_Wg = Hg // 2, Wg // 2

    out = np.zeros((Hf, Wf), dtype=f.dtype)
    f_padded = zero_pad(f, half_Hg, half_Wg)
    for i in range(Hf):
    	for j in range(Wf):
    		f_patch = np.ravel(f_padded[i:i+Hg, j:j+Wg])
    		f_patch_normalized = (f_patch - np.mean(f_patch)) / np.std(f_patch)
    		out[i, j] = np.dot(f_patch_normalized, np.ravel(g_normalized))
    ### END YOUR CODE

    return out
