"""
Implementation of the training routine for the 3D CNN with GAN
- train_dataset : list/array of 5D (or 5D ?) tensor in form (bs, input_channels, D_in, H_in, W_in)
"""
import torch.nn as nn
from torch.optim import Adadelta
import matplotlib.pyplot as plt
from IPython import display
from discriminator import Discriminator
from completion import CompletionN
from losses import completion_network_loss
from mean_pixel_value import MV_pixel
from utils import generate_input_mask, generate_hole_area, crop
from normalization import Normalization
from plot_error import Plot_Error
from get_dataset import *

num_channel = number_channel  # 0,1,2,3

path = 'result/model2015'  # result directory

train_dataset = get_list_model_tensor()

index_testing = -1

train_dataset, _, _ = Normalization(train_dataset)
testing_x = train_dataset[index_testing]  # test on the last element of the list
train_dataset.pop(index_testing)

mean_value_pixel = MV_pixel(train_dataset)  # compute the mean of the channel of the training set
mean_value_pixel = torch.tensor(mean_value_pixel.reshape(1, num_channel, 1, 1, 1))

# definitions of the hyperparameters
alpha = 4e-4
lr_c = 0.001
lr_d = 0.001
alpha = torch.tensor(alpha)
num_test_completions = 10
epoch1 = 250  # number of step for the first phase of training
epoch2 = 200  # number of step for the second phase of training
epoch3 = 500  # number of step for the third phase of training
snaperiod = 1
hole_min_d, hole_max_d = 10, 20
hole_min_h, hole_max_h = 30, 50
hole_min_w, hole_max_w = 30, 50
cn_input_size = (29, 65, 73)
ld_input_size = (20, 50, 50)

# make directory
path_configuration = path + '/' + str(epoch1) + 'ep1_' + str(epoch2) + 'ep2_' + str(epoch3) + 'ep3'
if not os.path.exists(path_configuration):
    os.mkdir(path_configuration)
path_lr = path_configuration + '/' + str(lr_c) + 'lrc_' + str(lr_d) + 'lrd'
if not os.path.exists(path_lr):
    os.mkdir(path_lr)

losses_1_c = []  # losses of the completion network during phase 1
losses_2_d = []  # losses of the discriminator network during phase 2
losses_3_c = []  # losses of the completion network during phase 3
losses_3_d = []  # losses of the discriminator network during phase 3

losses_1_c_test = []  # losses of TEST of the completion network during phase 1
losses_3_c_test = []  # losses of TEST of the completion network during phase 3

# PHASE 1
# COMPLETION NETWORK is trained with the MSE loss for T_c (=epoch1) iterations

model_completion = CompletionN()
optimizer_completion = Adadelta(model_completion.parameters(), lr=lr_c)
f = open(path_lr + "/phase1_losses.txt", "w+")
f_test = open(path_lr + "/phase1_TEST_losses.txt", "w+")
for ep in range(epoch1):
    for training_x in train_dataset:
        mask = generate_input_mask(
            shape=(training_x.shape[0], 1, training_x.shape[2], training_x.shape[3], training_x.shape[4]),
            hole_size=(hole_min_d, hole_max_d, hole_min_h, hole_max_h, hole_min_w, hole_max_w))
        training_x_masked = training_x - training_x * mask + mean_value_pixel * mask  # mask the training tensor with
        # pixel containing the mean value
        input = torch.cat((training_x_masked, mask), dim=1)
        output = model_completion(input.float())

        loss_completion = completion_network_loss(training_x, output, mask)  # MSE
        losses_1_c.append(loss_completion.item())

        print(f"[PHASE1 : EPOCH]: {ep + 1}, [LOSS]: {loss_completion.item():.12f}")
        display.clear_output(wait=True)
        f.write(f"[PHASE1 : EPOCH]: {ep + 1}, [LOSS]: {loss_completion.item():.12f} \n")

        optimizer_completion.zero_grad()
        loss_completion.backward()
        optimizer_completion.step()

    # test
    if ep % snaperiod == 0:
        model_completion.eval()
        with torch.no_grad():
            # testing_x = random.choice(test_dataset)
            training_mask = generate_input_mask(
                shape=(testing_x.shape[0], 1, testing_x.shape[2], testing_x.shape[3], testing_x.shape[4]),
                hole_size=(hole_min_d, hole_max_d, hole_min_h, hole_max_h, hole_min_w, hole_max_w),
                hole_area=generate_hole_area(ld_input_size,
                                             (training_x.shape[2], training_x.shape[3], training_x.shape[4])))
            testing_x_mask = testing_x - testing_x * training_mask + mean_value_pixel * training_mask
            testing_input = torch.cat((testing_x_mask, training_mask), dim=1)
            testing_output = model_completion(testing_input.float())

            loss_1c_test = completion_network_loss(testing_x, testing_output, training_mask)
            losses_1_c_test.append(loss_1c_test)

            print(f"[PHASE1 : EPOCH]: {ep + 1}, [TEST LOSS]: {loss_1c_test.item():.12f}")
            display.clear_output(wait=True)
            f_test.write(f"[PHASE1 : EPOCH]: {ep + 1}, [LOSS]: {loss_1c_test.item():.12f} \n")

            path_phase = path_lr + '/' + 'phase1'
            if not os.path.exists(path_phase):
                os.mkdir(path_phase)
            path_tensor = path_phase + '/tensor/'
            if not os.path.exists(path_tensor):
                os.mkdir(path_tensor)
            path_fig = path_phase + '/fig/'
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
                        plt.savefig(path_fig_channel + "/profondity_level_original_" + str(i) + ".png")
                        plt.close()

f.close()
f_test.close()

Plot_Error(losses_1_c_test, '1c', path_lr + '/')  # plot of the error in phase1

# PHASE 2
# COMPLETION NETWORK is FIXED and DISCRIMINATORS are trained form scratch for T_d (=epoch2) iterations

model_discriminator = Discriminator(loc_input_shape=(num_channel,) + ld_input_size,
                                    glo_input_shape=(num_channel,) + cn_input_size)
optimizer_discriminator = Adadelta(model_discriminator.parameters(), lr=lr_d)
loss_discriminator = nn.BCELoss()
f = open(path_lr + "/phase2_losses.txt", "w+")
for ep in range(epoch2):
    for training_x in train_dataset:
        # fake forward
        hole_area_fake = generate_hole_area(ld_input_size,
                                            (training_x.shape[2], training_x.shape[3], training_x.shape[4]))
        mask = generate_input_mask(
            shape=(training_x.shape[0], 1, training_x.shape[2], training_x.shape[3], training_x.shape[4]),
            hole_size=(hole_min_d, hole_max_d, hole_min_h, hole_max_h, hole_min_w, hole_max_w),
            hole_area=hole_area_fake)
        fake = torch.zeros((len(training_x), 1))
        training_x_masked = training_x - training_x * mask + mean_value_pixel * mask  # mask the training tensor with
        input_completion = torch.cat((training_x_masked, mask), dim=1)
        output_completion = model_completion(input_completion.float())
        input_global_discriminator_fake = output_completion.detach()
        input_local_discriminator_fake = crop(input_global_discriminator_fake, hole_area_fake)
        output_fake = model_discriminator((input_local_discriminator_fake, input_global_discriminator_fake))
        loss_fake = loss_discriminator(output_fake, fake)

        # real forward
        hole_area_real = generate_hole_area(ld_input_size,
                                            (training_x.shape[2], training_x.shape[3], training_x.shape[4]))
        real = torch.ones((len(training_x), 1))
        input_global_discriminator_real = training_x
        input_local_discriminator_real = crop(training_x, hole_area_real)
        output_real = model_discriminator((input_local_discriminator_real, input_global_discriminator_real))
        loss_real = loss_discriminator(output_real, real)

        loss = (loss_real + loss_fake) / 2.0
        losses_2_d.append(loss.item())

        print(f"[PHASE2 : EPOCH]: {ep + 1}, [LOSS]: {loss.item():.12f}")
        display.clear_output(wait=True)
        f.write(f"[PHASE2 : EPOCH]: {ep + 1}, [LOSS]: {loss.item():.12f} \n")

        loss.backward()
        optimizer_discriminator.step()
        optimizer_discriminator.zero_grad()

f.close()

Plot_Error(losses_2_d, '2d', path_lr + '/')

# PHASE 3
# both the completion network and content discriminators are trained jointly until the end of training

f = open(path_lr + "/phase3_com_losses.txt", "w+")
f_test = open(path_lr + "/phase3_TEST_losses.txt", "w+")
for ep in range(epoch3):
    for training_x in train_dataset:
        # fake forward
        hole_area_fake = generate_hole_area(ld_input_size,
                                            (training_x.shape[2], training_x.shape[3], training_x.shape[4]))
        mask = generate_input_mask(
            shape=(training_x.shape[0], 1, training_x.shape[2], training_x.shape[3], training_x.shape[4]),
            hole_size=(hole_min_d, hole_max_d, hole_min_h, hole_max_h, hole_min_w, hole_max_w),
            hole_area=hole_area_fake)
        fake = torch.zeros((len(training_x), 1))
        training_x_masked = training_x - training_x * mask + mean_value_pixel * mask
        input_completion = torch.cat((training_x_masked, mask), dim=1)
        output_completion = model_completion(input_completion.float())

        input_global_discriminator_fake = output_completion.detach()
        input_local_discriminator_fake = crop(input_global_discriminator_fake, hole_area_fake)
        output_fake = model_discriminator((input_local_discriminator_fake, input_global_discriminator_fake))
        loss_fake = loss_discriminator(output_fake, fake)

        # real forward
        hole_area_real = generate_hole_area(ld_input_size,
                                            (training_x.shape[2], training_x.shape[3], training_x.shape[4]))
        real = torch.ones((len(training_x), 1))
        input_global_discriminator_real = training_x
        input_local_discriminator_real = crop(training_x, hole_area_real)
        output_real = model_discriminator((input_local_discriminator_real, input_global_discriminator_real))
        loss_real = loss_discriminator(output_real, real)

        loss_d = (loss_real + loss_fake) * alpha / 2.
        losses_3_d.append(loss_d.item())

        # backward discriminator
        loss_d.backward()
        optimizer_discriminator.step()
        optimizer_discriminator.zero_grad()

        # forward completion
        loss_c1 = completion_network_loss(training_x, output_completion, mask)
        input_global_discriminator_fake = output_completion
        input_local_discriminator_fake = crop(input_global_discriminator_fake, hole_area_fake)
        output_fake = model_discriminator((input_local_discriminator_fake, input_global_discriminator_fake))
        loss_c2 = loss_discriminator(output_fake, real)

        loss_c = (loss_c1 + alpha * loss_c2) / 2.0
        losses_3_c.append(loss_c.item())

        # backward completion
        loss_c.backward()
        optimizer_completion.step()
        optimizer_completion.zero_grad()

        print(
            f"[PHASE3 : EPOCH]: {ep + 1}, [LOSS COMPLETION]: {loss_c.item():.12f}, [LOSS DISCRIMINATOR]: {loss_d.item():.12f}")
        display.clear_output(wait=True)
        f.write(
            f"[PHASE3 : EPOCH]: {ep + 1}, [LOSS COMPLETION]: {loss_c.item():.12f}, [LOSS DISCRIMINATOR]: {loss_d.item():.12f} \n")

    # test
    if ep % snaperiod == 0:
        model_completion.eval()
        with torch.no_grad():
            training_mask = generate_input_mask(
                shape=(testing_x.shape[0], 1, testing_x.shape[2], testing_x.shape[3], testing_x.shape[4]),
                hole_size=(hole_min_d, hole_max_d, hole_min_h, hole_max_h, hole_min_w, hole_max_w),
                hole_area=generate_hole_area(ld_input_size,
                                             (training_x.shape[2], training_x.shape[3], training_x.shape[4])))
            testing_x_mask = testing_x - testing_x * training_mask + mean_value_pixel * training_mask
            testing_input = torch.cat((testing_x_mask, training_mask), dim=1)
            testing_output = model_completion(testing_input.float())

            loss_3c_test = completion_network_loss(testing_x, testing_output, training_mask)
            losses_3_c_test.append(loss_3c_test)

            print(f"[PHASE3 : EPOCH]: {ep + 1}, [TEST LOSS]: {loss_3c_test.item():.12f}")
            display.clear_output(wait=True)
            f_test.write(f"[PHASE3 : EPOCH]: {ep + 1}, [LOSS]: {loss_3c_test.item():.12f} \n")

            path_phase = path_lr + '/' + 'phase3'
            if not os.path.exists(path_phase):
                os.mkdir(path_phase)
                path_tensor = path_phase + '/tensor/'
            if not os.path.exists(path_tensor):
                os.mkdir(path_tensor)
            path_fig = path_phase + '/fig/'
            if not os.path.exists(path_fig):
                os.mkdir(path_fig)

            path_tensor_epoch = path_tensor + 'epoch_' + str(ep)
            if not os.path.exists(path_tensor_epoch):
                os.mkdir(path_tensor_epoch)
            torch.save(testing_output, path_tensor_epoch + "/tensor_phase3" + ".pt")
            path_fig_epoch = path_fig + 'epoch_' + str(ep)
            if not os.path.exists(path_fig_epoch):
                os.mkdir(path_fig_epoch)

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

# save the model

path_model = 'model/model2015/model_completion_' + 'epoch_' + str(epoch1) + '_' + str(epoch2) + '_' + str(
    epoch3) + '_lrc_' + str(lr_c) + '_lrd_' + str(lr_d) + '.pt '
torch.save(model_completion.state_dict(), path_model)

f.close()

Plot_Error(losses_3_c_test, '3c', path_lr + '/')
Plot_Error(losses_3_d, '3d', path_lr + '/')

# printing specifics of the problem
print('epoch phase 1 : ', epoch1, ' -- epoch phase 2 : ', epoch2, ' -- epoch phase 3 : ', epoch3)
print('learning rate completion : ', lr_c, ' -- learning rate discriminator : ', lr_d)

# printing final loss training set
print('final loss of completion    network at phase 1 : ', losses_1_c[-1])
print('final loss of discriminator network at phase 2 : ', losses_2_d[-1])
print('final loss of completion    network at phase 3 : ', losses_3_c[-1])
print('final loss of discriminator network at phase 3 : ', losses_3_d[-1])

# printing final loss of testing set
print('final loss TEST at phase 1 : ', losses_1_c_test[-1])
print('final loss TEST at phase 3 : ', losses_3_c_test[-1])
