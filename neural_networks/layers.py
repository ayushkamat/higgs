"""
Author: Sophia Sanborn, Sagnik Bhattacharya
Institution: UC Berkeley
Date: Spring 2020
Course: CS189/289A
Website: github.com/sophiaas, github.com/sagnibak
"""

import numpy as np
from abc import ABC, abstractmethod

from neural_networks.activations import initialize_activation
from neural_networks.weights import initialize_weights
from neural_networks.utils.convolution import im2col, col2im, pad2d

from collections import OrderedDict

from typing import Callable, List, Tuple


class Layer(ABC):
    """Abstract class defining the `Layer` interface."""

    def __init__(self):
        self.activation = None

        self.n_in = None
        self.n_out = None

        self.parameters = {}
        self.cache = {}
        self.gradients = {}

        super().__init__()

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        pass

    def clear_gradients(self) -> None:
        self.cache = OrderedDict({a: [] for a, b in self.cache.items()})
        self.gradients = OrderedDict(
            {a: np.zeros_like(b) for a, b in self.gradients.items()}
        )

    def forward_with_param(
        self, param_name: str, X: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Call the `forward` method but with `param_name` as the variable with
        value `param_val`, and keep `X` fixed.
        """

        def inner_forward(param_val: np.ndarray) -> np.ndarray:
            self.parameters[param_name] = param_val
            return self.forward(X)

        return inner_forward

    def _get_parameters(self) -> List[np.ndarray]:
        return [b for a, b in self.parameters.items()]

    def _get_cache(self) -> List[np.ndarray]:
        return [b for a, b in self.cache.items()]

    def _get_gradients(self) -> List[np.ndarray]:
        return [b for a, b in self.gradients.items()]


def initialize_layer(
    name: str,
    activation: str = None,
    weight_init: str = None,
    n_out: int = None,
    kernel_shape: Tuple[int] = None,
    stride: int = None,
    pad: int = None,
    mode: str = None,
    keep_dim: str = "first",
) -> Layer:
    """Factory function for layers."""
    if name == "fully_connected":
        return FullyConnected(
            n_out=n_out, activation=activation, weight_init=weight_init,
        )

    elif name == "elman":
        return Elman(n_out=n_out, activation=activation, weight_init=weight_init,)

    elif name == "conv2d":
        return Conv2D(
            n_out=n_out,
            activation=activation,
            kernel_shape=kernel_shape,
            stride=stride,
            pad=pad,
            weight_init=weight_init,
        )

    else:
        raise NotImplementedError("Layer type {} is not implemented".format(name))


class FullyConnected(Layer):
    """A fully-connected layer multiplies its input by a weight matrix, adds
    a bias, and then applies an activation function.
    """

    def __init__(
        self, n_out: int, activation: str, weight_init="xavier_uniform"
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.activation = initialize_activation(activation)

        # instantiate the weight initializer
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int]) -> None:
        """Initialize all layer parameters (weights, biases)."""
        self.n_in = X_shape[1]

        ### BEGIN YOUR CODE ###

        W = self.init_weights((self.n_in, self.n_out))
        b = np.zeros((self.n_out))

        self.parameters = OrderedDict({"W": W, "b": b})
        self.cache: OrderedDict = OrderedDict({"Z": None, 'X': None})
        self.gradients: OrderedDict = OrderedDict({key: 0 for key in self.parameters.keys()})
                                      # MUST HAVE SAME KEYS AS `self.parameters`

        ### END YOUR CODE ###

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: multiply by a weight matrix, add a bias, apply activation.
        Also, store all necessary intermediate results in the `cache` dictionary
        to be able to compute the backward pass.

        Parameters
        ----------
        X  input matrix of shape (batch_size, input_dim)

        Returns
        -------
        a matrix of shape (batch_size, output_dim)
        """
        # initialize layer parameters if they have not been initialized
        if self.n_in is None:
            self._init_parameters(X.shape)

        ### BEGIN YOUR CODE ###

        W, b = self.parameters['W'], self.parameters['b']
        Z = X @ W + b
        self.cache['Z'] = Z
        self.cache['X'] = X
        out = self.activation.forward(Z)

        ### END YOUR CODE ###

        return out

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for fully connected layer.
        Compute the gradients of the loss with respect to:
            1. the weights of this layer (mutate the `gradients` dictionary)
            2. the bias of this layer (mutate the `gradients` dictionary)
            3. the input of this layer (return this)

        Parameters
        ----------
        dLdY  derivative of the loss with respect to the output of this layer
              shape (batch_size, output_dim)

        Returns
        -------
        derivative of the loss with respect to the input of this layer
        shape (batch_size, input_dim)
        """
        ### BEGIN YOUR CODE ###

        # unpack the cache
        
        Z, X, W = self.cache['Z'], self.cache['X'], self.parameters['W']
        self.gradients['b'] = np.sum(self.activation.backward(Z, dLdY), axis=0)
        self.gradients['W'] = X.T @ self.activation.backward(Z, dLdY)
        dX = self.activation.backward(Z, dLdY) @ W.T
        
        
        # compute the gradients of the loss w.r.t. all parameters as well as the
        # input of the layer

        # store the gradients in `self.gradients`
        # the gradient for self.parameters["W"] should be stored in
        # self.gradients["W"], etc.

        ### END YOUR CODE ###

        return dX


class Elman(Layer):
    """Elman recurrent layer."""

    def __init__(
        self,
        n_out: int,
        activation: str = "tanh",
        weight_init: str = "xavier_uniform",
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.activation = initialize_activation(activation)
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int]) -> None:
        """Initialize all layer parameters."""
        self.n_in = X_shape[1]

        ### BEGIN YOUR CODE ###

        W = self.init_weights((X_shape[1], self.n_out))
        U = self.init_weights((self.n_out, self.n_out))
        b = np.zeros(self.n_out)

        # initialize the cache, save the parameters, initialize gradients
        self.parameters: OrderedDict = OrderedDict({"W": W, "U": U, "b": b})
        self.gradients: OrderedDict = OrderedDict({key: None for key in self.parameters.keys()})

        ### END YOUR CODE ###

    def _init_cache(self, X_shape: Tuple[int]) -> None:
        """Initialize the layer cache. This contains useful information for
        backprop, crucially containing the hidden states.
        """
        ### BEGIN YOUR CODE ###

        s0 = np.zeros((X_shape[0], self.n_out))  # the first hidden state
        self.cache = OrderedDict({"s": [s0], "Z": [], "X": []})  # THIS IS INCOMPLETE

        ### END YOUR CODE ###

    def forward_step(self, X: np.ndarray) -> np.ndarray:
        """Compute a single recurrent forward step.
        Also, store all necessary intermediate results in the `cache` dictionary
        to be able to compute the backward pass.

        `self.cache["s"]` is a list storing all previous hidden states.
        The forward step is computed as:
            s_t+1 = fn(W X + U s_t + b)

        Parameters
        ----------
        X  input matrix of shape (batch_size, input_dim)

        Returns
        -------
        a matrix of shape (batch_size, output_dim)
        """
        ### BEGIN YOUR CODE ###

        # perform a recurrent forward step
        W, U, b, s = self.parameters['W'], self.parameters['U'], self.parameters['b'], self.cache['s'][-1]
        self.cache["X"].append(X)
        Z = X @ W
        Z += s @ U
        Z += b
        self.cache['Z'].append(Z)
        out = self.activation(Z)
        self.cache['s'].append(out)
        # store information necessary for backprop in `self.cache`

        ### END YOUR CODE ###

        return out

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the forward pass for `t` time steps. This should involve using
        forward_step repeatedly, possibly in a loop. This should be fairly simple
        since `forward_step` is doing most of the heavy lifting.

        Parameters
        ----------
        X  input matrix containing inputs for `t` time steps
           shape (batch_size, input_dim, t)

        Returns
        -------
        the final output/hidden state
        shape (batch_size, output_dim)
        """
        if self.n_in is None:
            self._init_parameters(X.shape[:2])

        self._init_cache(X.shape)

        ### BEGIN YOUR CODE ###
        Y = []
        # perform `t` forward passes through time and return the last
        # hidden/output state
        t = X.shape[-1]
        for i in range(t):
            Y.append(self.forward_step(X[:,:,i]))

        ### END YOUR CODE ###

        return Y[-1]

    def backward(self, dLdY: np.ndarray) -> List[np.ndarray]:
        """Backward pass for recurrent layer. Compute the gradient for all the
        layer parameters as well as every input at every time step.

        Parameters
        ----------
        dLdY  derivative of loss with respect to output of this layer
              shape (batch_size, output_dim)

        Returns
        -------
        list of numpy arrays of shape (batch_size, input_dim) of length `t`
        containing the derivative of the loss with respect to the input at each
        time step
        """
        ### BEGIN YOUR CODE ###

        # unpack the cache
        W, U, b, s = self.parameters['W'], self.parameters['U'], self.parameters['b'], self.cache['s']
        X, Z = self.cache["X"], self.cache["Z"]
        dLdX = []  # array containing dL / dx_i for all time steps i
        dLdW = np.zeros(W.shape)
        dLdU = np.zeros(U.shape)
        dLdb = np.zeros(b.shape)
        dS = [dLdY]  # array of form dL / ds_i for all time steps i
        t = len(self.cache['s'])
        # dS.append(dLdY)

        # perform backpropagation through time, storing the gradient of the loss
        # w.r.t. each time step in `dLdX`
        for i in range(1, t):
            si, si_1 = s[-i], s[-i - 1]
            ds_i = self.activation.backward(si_1, dS[0])
            dS.insert(0, ds_i @ U.T)
            dLdX.insert(0, ds_i @ W.T)

        for z, ds, x in zip(s[:-1], dS[1:], X):
            dLdW += x.T @ self.activation.backward(z, ds)
            dLdU += z.T @ self.activation.backward(z, ds)
            dLdb += np.sum(self.activation.backward(z, ds), axis=0)


        self.gradients["W"] = dLdW
        self.gradients["U"] = dLdU
        self.gradients["b"] = dLdb

        ### END YOUR CODE ###
        return dLdX


class Conv2D(Layer):
    """Convolutional layer for inputs with 2 spatial dimensions."""

    def __init__(
        self,
        n_out: int,
        kernel_shape: Tuple[int],
        activation: str,
        stride: int = 1,
        pad: str = "same",
        weight_init: str = "xavier_uniform",
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.pad = pad

        self.activation = initialize_activation(activation)
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int]) -> None:
        """Initialize all layer parameters."""
        ### BEGIN YOUR CODE ###

        # initialize weights, biases, the cache, and gradients
        self.n_in = X_shape[-1]
        W = self.init_weights((*self.kernel_shape, self.n_in, self.n_out))
        b = np.zeros((self.n_out))

        self.parameters: OrderedDict = OrderedDict({"W": W, "b": b})
        self.cache: OrderedDict = OrderedDict({})
        self.gradients: OrderedDict = OrderedDict({key: None for key in self.parameters.keys()})

        ### END YOUR CODE ###

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for convolutional layer. This layer convolves the input
        `X` with a filter of weights, adds a bias term, and applies an activation
        function to compute the output. This layer also supports padding and
        integer strides. Intermediates necessary for the backward pass are stored
        in the cache.

        Parameters
        ----------
        X  input with shape (batch_size, in_rows, in_cols, in_channels)

        Returns
        -------
        output feature maps with shape (batch_size, out_rows, out_cols, out_channels)
        """
        if self.n_in is None:
            self._init_parameters(X.shape)

        W = self.parameters["W"]
        b = self.parameters["b"]

        kernel_height, kernel_width, in_channels, out_channels = W.shape
        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_shape = (kernel_height, kernel_width)

        ### BEGIN YOUR CODE ###
        self.cache["X"] = X
        X, _ = pad2d(X, self.pad, kernel_shape, self.stride)
        # implement a convolutional forward pass
        Z = np.zeros((n_examples, X.shape[1] - kernel_height + 1, X.shape[2] - kernel_width + 1, self.n_out))
        rows, cols, examples, channels = np.arange(Z.shape[1]), np.arange(Z.shape[2]), np.arange(n_examples), np.arange(in_channels)
        for example in examples:
            x = X[example]
            z = np.zeros(Z.shape[1:])
            for row in rows:
                for col in cols:
                    x_slice = x[row: row + kernel_height, col: col + kernel_width, :]
                    r = np.tensordot(x_slice, W, axes=([0,1,2], [0,1,2])) + b
                    z[row, col] = r
            Z[example] = z
        self.cache["Z"] = Z
        out = self.activation(Z)

        # cache any values required for backprop

        ### END YOUR CODE ###

        return out

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for conv layer. Computes the gradients of the output
        with respect to the input feature maps as well as the filter weights and
        biases.

        Parameters
        ----------
        dLdY  derivative of loss with respect to output of this layer
              shape (batch_size, out_rows, out_cols, out_channels)

        Returns
        -------
        derivative of the loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, in_channels)
        """
        ### BEGIN YOUR CODE ###
        X = self.cache["X"]
        Z = self.cache["Z"]
        W = self.parameters["W"]
        b = self.parameters["b"]

        kernel_height, kernel_width, in_channels, out_channels = W.shape
        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_shape = (kernel_height, kernel_width)

        dldZ = self.activation.backward(Z, dLdY)
        dLdZ, p = pad2d(dldZ, self.pad, kernel_shape, self.stride)
        out = np.zeros(X.shape)
        rows, cols, examples = np.arange(out.shape[1]), np.arange(out.shape[2]), np.arange(n_examples)
        for example in examples:
            x = dLdZ[example]
            z = np.zeros(out.shape[1:])
            for row in rows:
                for col in cols:
                    x_slice = np.rot90(x[row: row + kernel_height, col: col + kernel_width, :], 2, (0, 1))
                    z[row, col] = np.tensordot(x_slice, W, axes=([0, 1, 2], [0, 1, 3]))
            out[example] = z
        dW = np.zeros(W.shape)
        for example in examples:
            x = X[example]
            dldz = dLdZ[example]
            z = np.zeros(kernel_shape)
            for o in range(self.n_out):
                for i in range(self.n_in):
                    im = x[:, :, i]
                    dz = dldz[:, :, o]
                    for row in range(kernel_height):
                        for col in range(kernel_width):
                            dz_slice = dz[kernel_height-row-1: kernel_height - row - 1 + im.shape[0] , kernel_width - col - 1: kernel_width - col - 1 + im.shape[1]]
                            z[row, col] = np.sum(im * dz_slice)
                    dW[:, :, i, o] += z
        self.gradients['W'] = dW
        self.gradients['b'] = np.sum(dldZ, axis=(0,1,2))
        ### END YOUR CODE ###

        return out

    def forward_faster(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for convolutional layer. This layer convolves the input
        `X` with a filter of weights, adds a bias term, and applies an activation
        function to compute the output. This layer also supports padding and
        integer strides. Intermediates necessary for the backward pass are stored
        in the cache.

        This implementation uses `im2col` which allows us to use fast general
        matrix multiply (GEMM) routines implemented by numpy. This is still
        rather slow compared to GPU acceleration, but still LEAGUES faster than
        the nested loop in the naive implementation.

        DO NOT ALTER THIS METHOD.

        You will write your naive implementation in forward().
        We will use forward_faster() to check your method.

        Parameters
        ----------
        X  input with shape (batch_size, in_rows, in_cols, in_channels)

        Returns
        -------
        output feature maps with shape (batch_size, out_rows, out_cols, out_channels)
        """
        if self.n_in is None:
            self._init_parameters(X.shape)

        W = self.parameters["W"]
        b = self.parameters["b"]

        kernel_height, kernel_width, in_channels, out_channels = W.shape
        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_shape = (kernel_height, kernel_width)

        X_col, p = im2col(X, kernel_shape, self.stride, self.pad)

        out_rows = int((in_rows + p[0] + p[1] - kernel_height) / self.stride + 1)
        out_cols = int((in_cols + p[2] + p[3] - kernel_width) / self.stride + 1)

        W_col = W.transpose(3, 2, 0, 1).reshape(out_channels, -1)

        Z = (
            (W_col @ X_col)
            .reshape(out_channels, out_rows, out_cols, n_examples)
            .transpose(3, 1, 2, 0)
        )
        Z += b
        out = self.activation(Z)

        self.cache["Z"] = Z
        self.cache["X"] = X

        return out

    def backward_faster(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for conv layer. Computes the gradients of the output
        with respect to the input feature maps as well as the filter weights and
        biases.

        This uses im2col, so it is considerably faster than the naive implementation
        even on a CPU.

        DO NOT ALTER THIS METHOD.

        You will write your naive implementation in backward().
        We will use backward_faster() to check your method.

        Parameters
        ----------
        dLdY  derivative of loss with respect to output of this layer
              shape (batch_size, out_rows, out_cols, out_channels)

        Returns
        -------
        derivative of the loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, in_channels)
        """
        W = self.parameters["W"]
        b = self.parameters["b"]
        Z = self.cache["Z"]
        X = self.cache["X"]

        kernel_height, kernel_width, in_channels, out_channels = W.shape
        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_shape = (kernel_height, kernel_width)

        dZ = self.activation.backward(Z, dLdY)

        dZ_col = dZ.transpose(3, 1, 2, 0).reshape(dLdY.shape[-1], -1)
        X_col, p = im2col(X, kernel_shape, self.stride, self.pad)
        W_col = W.transpose(3, 2, 0, 1).reshape(out_channels, -1).T

        dW = (
            (dZ_col @ X_col.T)
            .reshape(out_channels, in_channels, kernel_height, kernel_width)
            .transpose(2, 3, 1, 0)
        )
        dB = dZ_col.sum(axis=1).reshape(1, -1)

        dX_col = W_col @ dZ_col
        dX = col2im(dX_col, X, W.shape, self.stride, p).transpose(0, 2, 3, 1)

        self.gradients["W"] = dW
        self.gradients["b"] = dB

        return dX
