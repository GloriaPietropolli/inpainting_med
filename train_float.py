"""
The goal is to take the model already trained with model data and train it again with float values
using a weight matrix to perform training only where floating information are available
"""
from dumb_list import *
from normalization import Normalization
from utils import *
from get_dataset import *
from completion import CompletionN
from losses import completion_network_loss
from mean_pixel_value import *
import matplotlib.pyplot as plt

# first of all we get the model trained with model's data
path_model = 'model/' + kindof + '/'
list_avaiable_models = os.listdir(path_model)


