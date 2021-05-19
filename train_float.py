"""
The goal is to take the model already trained with model data and train it again with float values
using a weight matrix to perform training only where floating information are available
"""
import matplotlib.pyplot as plt
from torch.optim import Adadelta
from IPython import display
from normalization import Normalization, Normalization_Float
from completion import CompletionN
from utils import *
from get_dataset import *
from losses import completion_float_loss
from mean_pixel_value import *
from plot_error import Plot_Error
from float_mask import *


# first of all we get the model trained with model's data
path_model = 'model/model2015/'
list_avaiable_models = os.listdir(path_model)
select_model = 'model_completion_epoch_250_150_150_lrc_0.01_lrd_0.01.pt '
name_model = select_model[:-3]
print('model used : ', name_model)

model_completion = CompletionN()
model_completion.load_state_dict(torch.load(path_model + select_model))
model_completion.eval()

path = 'result2/' + name_model + '/'  # where we save the information
if not os.path.exists(path):
    os.mkdir(path)

weight_float = get_list_float_weight_tensor()
data_float = get_list_float_tensor()
data_model = get_list_model_tensor()

test_dataset, mean_tensor, std_tensor = Normalization(data_model)
train_dataset = Normalization_Float(data_float, mean_tensor, std_tensor)

for el in range(len(train_dataset)):
    train_dataset[el] = train_dataset[el][:, :, :-1, :, :]
for el in range(len(weight_float)):
    weight_float[el] = weight_float[el][:, :, 1:-1, :, 1:-1]

index_test = -1
testing_x = test_dataset[index_test]
testing_weight = weight_float[index_test]

mean_value_pixel = MV_pixel(test_dataset)
mean_value_pixel = torch.tensor(mean_value_pixel.reshape(1, 4, 1, 1, 1))

# parameters for the second train routine
lr_c = 0.00001
epoch1 = 50  # number of step for the first phase of training
snaperiod = 1
hole_min_d, hole_max_d = 5, 10
hole_min_h, hole_max_h = 10, 20
hole_min_w, hole_max_w = 10, 20
cn_input_size = (29, 65, 73)
ld_input_size = (20, 50, 50)

path_configuration = path + '/' + str(epoch1)
if not os.path.exists(path_configuration):
    os.mkdir(path_configuration)
path_lr = path_configuration + '/' + str(lr_c)
if not os.path.exists(path_lr):
    os.mkdir(path_lr)

losses_1_c = []  # losses of the completion network during phase 1
losses_1_c_test = []  # losses of TEST of the completion network during phase 1

# COMPLETION NETWORK is trained with the MSE loss for T_c (=epoch1) iterations
optimizer_completion = Adadelta(model_completion.parameters(), lr=lr_c)
f = open(path_lr + "/phase1_losses.txt", "w+")
f_test = open(path_lr + "/phase1_TEST_losses.txt", "w+")
for ep in range(epoch1):
    for i in range(len(train_dataset)):
        training_x = train_dataset[i]
        weight = weight_float[i]

        mask = generate_input_mask(
            shape=(training_x.shape[0], 1, training_x.shape[2], training_x.shape[3], training_x.shape[4]),
            hole_size=(hole_min_d, hole_max_d, hole_min_h, hole_max_h, hole_min_w, hole_max_w))

        mask = make_float_mask(weight, mask)

        training_x_masked = training_x - training_x * mask + mean_value_pixel * mask
        input = torch.cat((training_x_masked, mask), dim=1)
        output = model_completion(input.float())

        loss_completion = completion_float_loss(training_x, output, mask)  # MSE
        losses_1_c.append(loss_completion.item())

        print(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_completion.item():.5e}")
        display.clear_output(wait=True)
        f.write(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_completion.item():.5e} \n")

        if loss_completion == 0:
            print('non eseguo il training')
            continue

        optimizer_completion.zero_grad()
        loss_completion.backward()
        optimizer_completion.step()

    # test
    if ep % snaperiod == 0:
        model_completion.eval()
        with torch.no_grad():
            testing_mask = generate_input_mask(
                shape=(testing_x.shape[0], 1, testing_x.shape[2], testing_x.shape[3], testing_x.shape[4]),
                hole_size=(hole_min_d, hole_max_d, hole_min_h, hole_max_h, hole_min_w, hole_max_w),
                hole_area=generate_hole_area(ld_input_size,
                                             (testing_x.shape[2], testing_x.shape[3], testing_x.shape[4])))

            testing_mask = make_float_mask(testing_weight, testing_mask)

            testing_x_mask = testing_x - testing_x * testing_mask + mean_value_pixel * testing_mask
            testing_input = torch.cat((testing_x_mask, testing_mask), dim=1)
            testing_output = model_completion(testing_input.float())

            loss_1c_test = completion_float_loss(testing_x, testing_output, testing_mask)
            losses_1_c_test.append(loss_1c_test.item())

            print(f"[EPOCH]: {ep + 1}, [TEST LOSS]: {loss_1c_test.item():.5f}")
            display.clear_output(wait=True)
            f_test.write(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_1c_test.item():.5f} \n")

            path_tensor = path_lr + '/tensor/'
            if not os.path.exists(path_tensor):
                os.mkdir(path_tensor)
            path_fig = path_lr + '/fig/'
            if not os.path.exists(path_fig):
                os.mkdir(path_fig)

            path_tensor_epoch = path_tensor + 'epoch_' + str(ep)
            if not os.path.exists(path_tensor_epoch):
                os.mkdir(path_tensor_epoch)
            torch.save(testing_output, path_tensor_epoch + "/tensor_phase1" + ".pt")

            path_fig_epoch = path_fig + 'epoch_' + str(ep)
            if not os.path.exists(path_fig_epoch):
                os.mkdir(path_fig_epoch)

            path_fig_original = path_fig + 'original_fig'
            if not os.path.exists(path_fig_original):
                os.mkdir(path_fig_original)

            number_fig = len(testing_output[0, 0, :, 0, 0])  # number of levels of depth

            for channel in channels:
                for i in range(number_fig):
                    path_fig_channel = path_fig_epoch + '/' + str(channel)
                    if not os.path.exists(path_fig_channel):
                        os.mkdir(path_fig_channel)
                    cmap = plt.get_cmap('Greens')
                    plt.imshow(testing_output[0, channel, i, :, :], cmap=cmap)
                    plt.colorbar()
                    plt.savefig(path_fig_channel + "/profondity_level_" + str(i) + ".png")
                    plt.close()

                    if ep == 0:
                        path_fig_channel = path_fig_original + '/' + str(channel)
                        if not os.path.exists(path_fig_channel):
                            os.mkdir(path_fig_channel)
                        plt.imshow(testing_x[0, channel, i, :, :], cmap=cmap)
                        plt.colorbar()
                        plt.savefig(path_fig_channel + "/profundity_level_original_" + str(i) + ".png")
                        plt.close()

f.close()
f_test.close()
Plot_Error(losses_1_c_test, '1c', path_lr + '/')  # plot of the error in phase1


# printing final loss training set
print('final loss TRAINING : ', losses_1_c[-1])

# printing final loss of testing set
print('final loss TEST : ', losses_1_c_test[-1])

print('model used : ', name_model)
print('learning rate used for the float data training : ', lr_c)
