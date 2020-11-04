from tensorflowcv.model_provider import get_model as tfcv_get_model
from tensorflowcv.model_provider import init_variables_from_state_dict as tfcv_init_variables_from_state_dict
import tensorflow as tf
import numpy as np

net = tfcv_get_model("resnet18", pretrained=True, data_format="channels_first")
net.summary()
