'''
This simulation executes a trained neural network that approximates the
closed form solution given by 2 axis inv kin
'''
from __future__ import division

import numpy as np
import contextlib

with contextlib.redirect_stdout(None):
    pass
import sys
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import helpers
# from neural_network import FullyConnectedNetwork
from neural_network_solution import FullyConnectedNetwork

'''Global Variables'''
batch_sz = 2056
input_shape = 2
output_shape = 4
drop_rte = 0.1
hidden_neurons = [40, 40, 40, 40, output_shape]
model = FullyConnectedNetwork(input_shape, hidden_neurons, drop_rte)
normalize_data_bool = True


def inv_kin_2arm(x, y, link0_length, link1_length):
    theta_0 = 0
    theta_1 = 0
    return theta_0, theta_1


def load_model(model):
    # return torch.load('./saved_models/deterministicmodel.pth')
    # return torch.load('./saved_models/mysavedmodel.pth')
    return model.load_state_dict(torch.load('./mysavedmodel.pth'))


def normalize(arr, normalize_type):
    # Type 0 is between 0 and pi
    if normalize_type == 0:
        for i in range(int(arr.size)):
            arr[i] = (arr[i]) / (np.pi)
        return arr
    elif normalize_type == 1:  # Type 1 is between 0 and 2 pi
        for i in range(int(arr.size)):
            arr[i] = (arr[i]) / (2 * np.pi)
        return arr
    else:  # Type 2 is between -420 and 420
        for i in range(int(arr.size)):
            arr[i] = (arr[i] - (-420)) / (840)
        return arr


'''Test Neural Network'''


def test_model(data, label):
    global model
    global batch_sz

    # Prepare model for testing
    model.eval()
    test_loss = 0
    correct = 0
    test_steps = 0
    mseloss = nn.MSELoss()

    for batch_idx in range(int(data.shape[0] / batch_sz)):
        if torch.cuda.is_available():
            data_batch = Variable(
                torch.from_numpy(data[batch_idx * batch_sz:(batch_idx + 1) * batch_sz]).float().cuda())
            label_batch = Variable(
                torch.from_numpy(label[batch_idx * batch_sz:(batch_idx + 1) * batch_sz]).float().cuda())
        else:
            data_batch = Variable(torch.from_numpy(data[batch_idx * batch_sz:(batch_idx + 1) * batch_sz]).float())
            label_batch = Variable(torch.from_numpy(label[batch_idx * batch_sz:(batch_idx + 1) * batch_sz]).float())

        # Inference step followed by loss calculation
        output = model(data_batch)
        test_loss += mseloss(output, label_batch).item()  # sum up batch loss
        test_steps += 1
        for i in range(batch_sz):
            if normalize_data_bool:
                x = data[batch_idx * batch_sz + i][0] * 840 - 420
                y = data[batch_idx * batch_sz + i][1] * 840 - 420
            else:
                x = data[batch_idx * batch_sz + i][0]
                y = data[batch_idx * batch_sz + i][1]
            theta_0 = np.arctan2(output[i][0].item(), output[i][1].item())
            theta_1 = np.arctan2(output[i][2].item(), output[i][3].item())
            theta_0_label = np.arctan2(label_batch[i][0].item(), label_batch[i][1].item())
            theta_1_label = np.arctan2(label_batch[i][2].item(), label_batch[i][3].item())
            theta_0_true, theta_1_true = inv_kin_2arm(x, y, 186, 269 - 35)
            theta_0, theta_1 = helpers.convert_normal_angle(theta_0, theta_1)
            theta_0_label, theta_1_label = helpers.convert_normal_angle(theta_0_label, theta_1_label)
            theta_0_true, theta_1_true = helpers.convert_normal_angle(theta_0_true, theta_1_true)
            print("error={} x={} y={} theta_0={:.3f} theta_1={:.3f} theta_0_true={:.3f} theta_1_true={:.3f}".format(
                np.sqrt((theta_0 - theta_0_true) ** 2 + (theta_1 - theta_1_true) ** 2),
                x, y, theta_0, theta_1, theta_0_true, theta_1_true,
            ))

    print('\nTest set: Average loss: {:.4f}'.format(test_loss / test_steps))
    return (test_loss / test_steps)


load_model(model)
print(model)
'''Move Model to GPU if Cuda'''
if torch.cuda.is_available():
    model.cuda()
    print("Using GPU Acceleration")
else:
    print("Not Using GPU Acceleration")

'''Load Data'''
dir_path = os.path.dirname(os.path.realpath('inv_kin_closed_form_arm.py'))
train = np.load(dir_path + '/data/inv_kin_aprox/train.npy')
test = np.load(dir_path + '/data/inv_kin_aprox/train.npy')
best_loss = float(sys.maxsize)
if normalize_data_bool:
    print("Normalizing Input and Output")

    train[:, 0] = normalize(train[:, 0], 2)
    train[:, 1] = normalize(train[:, 1], 2)

    test[:, 0] = normalize(test[:, 0], 2)
    test[:, 1] = normalize(test[:, 1], 2)

# Shuffle data
np.random.shuffle(train)
data = train[:, :2]
label = train[:, 2:]

np.random.shuffle(test)
test_data = test[:, :2]
test_label = test[:, 2:]

with torch.no_grad():
    test_model(test_data, test_label)
