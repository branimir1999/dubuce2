from typing import Callable, Union

import tensorflow as tf

TFData = Union[tf.Tensor, tf.Variable, float]
TFBuiltin = Union[Callable, str]
