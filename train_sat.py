"""
The goal is to take the model already trained with model data and train it again with float values
using a weight matrix to perform training only where floating information are available
"""
import matplotlib.pyplot as plt
from torch.optim import Adadelta
from IPython import display
from float_mask import *
from normalization import Normalization, Normalization_Sat
from completion import CompletionN
from utils import *
from get_dataset import *
from losses import completion_sat_loss
from mean_pixel_value import *
from plot_error import Plot_Error

# first of all we get the model trained with model's data
path_model = 'model/model2015/'
list_avaiable_models = os.listdir(path_model)
select_model = 'model_completion_epoch_250_150_150_lrc_0.01_lrd_0.01.pt '
name_model = select_model[:-3]
print('model used : ', name_model)

model_completion = CompletionN()
model_completion.load_state_dict(torch.load(path_model + select_model))
model_completion.eval()

path = 'result3/' + name_model + '/'  # where we save the information
if not os.path.exists(path):
    os.mkdir(path)

weight_sat = get_list_sat_weight_tensor()
data_sat = get_list_sat_tensor()
data_model = get_list_model_tensor()

test_dataset, mean_tensor, std_tensor = Normalization(data_model)
train_dataset = Normalization_Sat(data_sat, mean_tensor, std_tensor)

for el in range(len(train_dataset)):
    train_dataset[el] = train_dataset[el][:, :, :, :, :]
for el in range(len(weight_sat)):
    weight_sat[el] = weight_sat[el][:, :, 0:-2, :, 1:-1]

index_test = -1
testing_x = train_dataset[index_test]
testing_x_model = test_dataset[index_test]
testing_weight = weight_sat[index_test]


mean_value_pixel = MV_pixel(train_dataset)  # compute the mean of the channel of the training set
mean_value_pixel = torch.tensor(mean_value_pixel.reshape(1, 4, 1, 1, 1))

lr_c = 1e-04
epoch1 = 50  # number of step for the first phase of training
snaperiod = 1
hole_min_d, hole_max_d = 28, 29
hole_min_h, hole_max_h = 30, 50
hole_min_w, hole_max_w = 30, 50
cn_input_size = (29, 65, 73)
ld_input_size = (20, 50, 50)

path_configuration = path + '/' + str(epoch1)
if not os.path.exists(path_configuration):
    os.mkdir(path_configuration)
path_lr = path_configuration + '/' + str(lr_c)
if not os.path.exists(path_lr):
    os.mkdir(path_lr)

losses = []  # losses of the completion network
losses_test = []  # losses of TEST of the completion network

# COMPLETION NETWORK is trained with the MSE loss for T_c (=epoch1) iterations
optimizer_completion = Adadelta(model_completion.parameters(), lr=lr_c)
f = open(path_lr + "/phase1_losses.txt", "w+")
f_test = open(path_lr + "/phase1_TEST_losses.txt", "w+")
f_test_model = open(path_lr + "/phase1_TEST_MODEL_losses.txt", "w+")
for ep in range(epoch1):
    for i in range(len(train_dataset)):
        training_sat = train_dataset[i]
        training_x = test_dataset[i]
        weight = weight_sat[i]

        training_x[:, 3:, 0:2, :, :] = training_sat[:, 3:, 0:1, :, :]

        mask = generate_input_mask(
            shape=(training_x.shape[0], 1, training_x.shape[2], training_x.shape[3], training_x.shape[4]),
            hole_size=(hole_min_d, hole_max_d, hole_min_h, hole_max_h, hole_min_w, hole_max_w))

        training_x_masked = training_x - training_x * mask + mean_value_pixel * mask
        # training_x_masked = masked_sat_tensor(training_x.clone().detach(), mean_value_pixel, mask)

        input = torch.cat((training_x_masked, mask), dim=1)
        output = model_completion(input.float())

        mask = make_sat_mask(weight, mask)
        loss_completion = completion_sat_loss(training_x, output, mask)  # MSE
        losses.append(loss_completion.item())

        print(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_completion.item():.5e}")
        display.clear_output(wait=True)
        f.write(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_completion.item():.5e} \n")

        if loss_completion == 0:
            print('non eseguo il training')
            continue

        optimizer_completion.zero_grad()
        loss_completion.backward()
        optimizer_completion.step()

    # test on sat data
    if ep % snaperiod == 0:
        model_completion.eval()
        with torch.no_grad():
            testing_mask = generate_input_mask(
                shape=(testing_x.shape[0], 1, testing_x.shape[2], testing_x.shape[3], testing_x.shape[4]),
                hole_size=(hole_min_d, hole_max_d, hole_min_h, hole_max_h, hole_min_w, hole_max_w))  # DELETE

            testing_mask = make_sat_mask(testing_weight, testing_mask)

            testing_x_masked = masked_sat_tensor(testing_x.clone().detach(), mean_value_pixel, testing_mask)

            testing_input = torch.cat((testing_x_masked, testing_mask), dim=1)
            testing_output = model_completion(testing_input.float())

            loss_test = completion_sat_loss(testing_x, testing_output, testing_mask)
            losses_test.append(loss_test.item())

            print(f"[EPOCH]: {ep + 1}, [TEST LOSS]: {loss_test.item():.5e}")
            display.clear_output(wait=True)
            f_test.write(f"[EPOCH]: {ep + 1}, [LOSS]: {loss_test.item():.5e} \n")

            # test on model data (how the input image change)
            if ep % snaperiod == 0:
                model_completion.eval()
                with torch.no_grad():
                    testing_mask_model = generate_input_mask(
                        shape=(testing_x.shape[0], 1, testing_x.shape[2], testing_x.shape[3], testing_x.shape[4]),
                        hole_size=(hole_min_d, hole_max_d, hole_min_h, hole_max_h, hole_min_w, hole_max_w))  # DELETE

                    testing_x_mask = testing_x_model - testing_x_model * testing_mask_model + mean_value_pixel * testing_mask_model
                    testing_input = torch.cat((testing_x_mask, testing_mask_model), dim=1)
                    testing_output = model_completion(testing_input.float())

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
                                plt.imshow(testing_x_model[0, channel, i, :, :], cmap=cmap)
                                plt.colorbar()
                                plt.savefig(path_fig_channel + "/profundity_level_original_" + str(i) + ".png")
                                plt.close()


path_model = 'model/sat/model_sat_' + 'epoch_' + str(epoch1) + '_lr_' + str(lr_c) + '.pt '
torch.save(model_completion.state_dict(), path_model)

f.close()
f_test.close()

Plot_Error(losses_test, 'sat', path_lr + '/')  # plot of the error on sat data

# printing final loss training set
print('final loss TRAINING : ', losses[-1])

# printing final loss of testing set
print('final loss TEST : ', losses_test[-1])

print('model used : ', name_model)
print('learning rate used for the float data training : ', lr_c)
