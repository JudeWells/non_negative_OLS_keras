import tensorflow as tf
import keras
import numpy as np
from keras import backend as K
import sys

class Constraint(object):
    """Constraint base class:
    Function that imposes constraints on weight values
    """

    def __call__(self, w):
        return w

    def get_config(self):
        return {}

class NonNegSumToOne(Constraint):
    """Constrains all weights to be greater than zero.
    Ensures that all weights sum to 1
    """
    def __init__(self, axis=0):
        self.axis = axis

    def get_config(self):
        return {'axis': self.axis}

    def __call__(self, w):
        NonNeg = w * K.cast(K.greater_equal(w, 0.), K.floatx())
        return NonNeg / (K.epsilon() + K.sum(NonNeg))

class Regularizer(object):
    """Regularizer base class.
    """

    def __call__(self, x):
        return 0.

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class l1_and_price(Regularizer):
    """Regularizer for L1 regularization and price weighting.
    price weighting adds loss for each weight propostional to
    the price associated with each weight.

    # Arguments
        l1: Float; L1 regularization factor.
        price_weight; weighting to apply to added price loss
        prices; a tf.tensor of shape (1, num_weights, 1)
    """

    def __init__(self, l1=0.1, price_weight=0, prices=tf.constant(0., shape=(1,16,1))):
        self.l1 = K.cast_to_floatx(l1)
        self.prices = prices
        self.price_weight = price_weight

    def __call__(self, x):
        regularization = 0.
        regularization += self.l1 * K.sum(K.abs(x))
        regularization += self.price_weight * K.sum(tf.tensordot(self.prices, x, axes=2))
        return regularization

class nn_ols():
    """Initialises and trains OLS model with data and hyper-parameters provided during initialisation
    """
    def __init__(self, X, y, l1=0, price_weight=0, prices=tf.constant(0., shape=(1,16,1)), lr=0.001, epochs=250):
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(1,
                                     input_dim=X.shape[1],
                                     activation='linear',
                                     use_bias=False,
                                     kernel_constraint=NonNegSumToOne(),
                                     kernel_regularizer = l1_and_price(l1=0, price_weight=0, prices=prices),
                                     kernel_initializer= keras.initializers.Constant(value=0.1)))

        sgd = keras.optimizers.SGD(lr=0.0005, decay=1e-6, momentum=0.1, nesterov=True)
        self.model.compile(loss='mean_squared_error', optimizer=sgd)

        self.model.fit(X, y, epochs = epochs, batch_size=len(X), verbose = 0)
        self.weights = self.model.weights
        self.non_zero_ingredients = K.sum(tf.cast(K.greater_equal(self.model.weights, 0.01), tf.int32)).numpy()
