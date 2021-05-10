from dumb_list import *
from normalization import Normalization
from utils import *
from get_dataset import *
from completion import CompletionN
from losses import completion_network_loss
from mean_pixel_value import *
# how to get a model

path_model = 'model/' + kindof + '/'
list_avaiable_models = os.listdir(path_model)
a_model = list_avaiable_models[0]

model = CompletionN()
model.load_state_dict(torch.load(path_model + a_model))
model.eval()

path = 'result/' + kindof  # result directory

if kindof == 'float':
    train_dataset = list_float_tensor
if kindof == 'model2015':
    train_dataset = list_model_tensor
if kindof == 'sat':
    train_dataset = list_sat_tensor
if kindof == 'dumb':
    train_dataset = dumb_dataset

train_dataset = Normalization(train_dataset)
testing_x = train_dataset[20]

alpha = 4e-4
lr_c = 0.001
lr_d = 0.001
alpha = torch.tensor(alpha)
num_test_completions = 10
epoch1 = 100  # number of step for the first phase of training
epoch2 = 25  # number of step for the second phase of training
epoch3 = 100  # number of step for the third phase of training
snaperiod = 1
hole_min_d, hole_max_d = 10, 20
hole_min_h, hole_max_h = 30, 50
hole_min_w, hole_max_w = 30, 50
cn_input_size = (29, 65, 73)
ld_input_size = (20, 50, 50)

mean_value_pixel = MV_pixel(train_dataset)  # compute the mean of the channel of the training set
mean_value_pixel = torch.tensor(mean_value_pixel.reshape(1, 4, 1, 1, 1))  # transform the mean_value_pixel
# (an array of length 3) into a tensor of the same shape as the input's ones

with torch.no_grad():
    # testing_x = random.choice(test_dataset)
    training_mask = generate_input_mask(
        shape=(testing_x.shape[0], 1, testing_x.shape[2], testing_x.shape[3], testing_x.shape[4]),
        hole_size=(hole_min_d, hole_max_d, hole_min_h, hole_max_h, hole_min_w, hole_max_w),
        hole_area=generate_hole_area(ld_input_size,
                                     (testing_x.shape[2], testing_x.shape[3], testing_x.shape[4])))
    testing_x_mask = testing_x - testing_x * training_mask + mean_value_pixel * training_mask
    testing_input = torch.cat((testing_x_mask, training_mask), dim=1)
    testing_output = model(testing_input.float())

    loss_1c_test = completion_network_loss(testing_x, testing_output, testing_x_mask)
    print(loss_1c_test)
