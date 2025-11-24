from __future__ import division

import argparse
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import configparser

'''Baseline Dataset'''
from data.sequential_mnist import SequentialMNIST

'''Models of Interest'''
from models.lstm_solution import LSTM
from models.irnn import IRNN
from models.gru_solution import GRU
from models.peephole_lstm import Peephole
from models.ugrnn import UGRNN
from models.intersection_rnn import IntersectionRNN
from models.rnn_solution import RNN

'''Tensorboard imports'''
# DO NOT FORGET TO RUN TENSORBOARD IN YOUR TERMINAL
#  tensorboard --logdir=exercise4_RNN/rnn_zoo_incomplete/runs
#
from torch.utils.tensorboard import SummaryWriter
import os
import socket
from datetime import datetime

config = configparser.ConfigParser()
config.read('config.ini')

base_dir = config['DEFAULT']['BASE_DIR']
sys.path.append(base_dir + '/models/')
sys.path.append(base_dir + '/data/')

# Define global variables and arguments for experimentation
parser = argparse.ArgumentParser(description='Nueron Connection')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='learning rate (default: 1e-4)')
parser.add_argument('--epochs', type=int, default=100,
                    help='num training epochs (default: 75)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--hx', type=int, default=100,
                    help='hidden vec size for rnn models (default: 100)')
parser.add_argument('--layers', type=int, default=1,
                    help='num recurrent layers (default: 1)')
parser.add_argument('--batch-size', type=int, default=64,
                    help='training batch size (default: 64)')
parser.add_argument('--model-type', type=str, default='lstm',
                    help='rnn, lstm, gru, irnn, ugrnn, rnn+, peephole')
parser.add_argument('--task', type=str, default='seqmnist',
                    help='seqmnist, pseqmnist')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use gpu')
parser.add_argument('--drop', type=float, default=0,
                    help='Drop input connections in input weight matrix (Default:0)')
parser.add_argument('--rec_drop', type=float, default=0,
                    help='Drop recurrent connections in recurrent weight matrix (Default:0)')

args = parser.parse_args()

# Check if cuda is available
args.cuda = args.cuda and torch.cuda.is_available()
if args.cuda:
    print("Using GPU Acceleration")

# Set experimentation seed
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

np.random.seed(args.seed)
random.seed(args.seed)


# Define helper function
def log_sigmoid(x):
    return torch.log(torch.sigmoid(x))


args.task = 'seqmnist'

# Load data loader depending on task of interest
if args.task == 'seqmnist':
    print("Loading SeqMNIST")
    dset = SequentialMNIST()
else:
    args.task = 'pseqmnist'
    print("Loading PSeqMNIST")
    dset = SequentialMNIST(permute=True)

data_loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, num_workers=2, shuffle=True)
activation = nn.LogSoftmax(dim=1)
criterion = nn.CrossEntropyLoss()


# Define deep recurrent neural network
def create_model():
    if args.model_type == 'lstm':
        return LSTM(input_size=dset.input_dimension,
                    hidden_size=args.hx,
                    output_size=dset.output_dimension,
                    layers=args.layers,
                    drop=args.drop,
                    rec_drop=args.rec_drop)
    elif args.model_type == 'rnn':
        return RNN(input_size=dset.input_dimension,
                   hidden_size=args.hx,
                   output_size=dset.output_dimension,
                   layers=args.layers,
                   drop=args.drop,
                   rec_drop=args.rec_drop)
    elif args.model_type == 'irnn':
        return IRNN(input_size=dset.input_dimension,
                    hidden_size=args.hx,
                    output_size=dset.output_dimension,
                    layers=args.layers,
                    drop=args.drop,
                    rec_drop=args.rec_drop)
    elif args.model_type == 'gru':
        return GRU(input_size=dset.input_dimension,
                   hidden_size=args.hx,
                   output_size=dset.output_dimension,
                   layers=args.layers,
                   drop=args.drop,
                   rec_drop=args.rec_drop)
    elif args.model_type == 'rnn+':
        if args.layers == 1:
            args.layers = 2
        return IntersectionRNN(input_size=dset.input_dimension,
                               hidden_size=args.hx,
                               output_size=dset.output_dimension,
                               layers=args.layers,
                               drop=args.drop,
                               rec_drop=args.rec_drop)
    elif args.model_type == 'peephole':
        return Peephole(input_size=dset.input_dimension,
                        hidden_size=args.hx,
                        output_size=dset.output_dimension,
                        layers=args.layers,
                        drop=args.drop,
                        rec_drop=args.rec_drop)
    elif args.model_type == 'ugrnn':
        return UGRNN(input_size=dset.input_dimension,
                     hidden_size=args.hx,
                     output_size=dset.output_dimension,
                     layers=args.layers,
                     drop=args.drop,
                     rec_drop=args.rec_drop)
    else:
        raise Exception


# Execute neural network on entire input sequence
def execute_sequence(seq, target, model):
    predicted_list = []
    y_list = []
    model.reset(batch_size=seq.size(0), cuda=args.cuda)

    for i, input_t in enumerate(seq.chunk(seq.size(1), dim=1)):
        input_t = input_t.squeeze(1)
        if activation is None:
            p = model(input_t)
        else:
            p = model(input_t)
            p = activation(p)
        predicted_list.append(p)
        y_list.append(target)

    return predicted_list, y_list


# Train neural network
def train(epoch, model, optimizer):
    dset.train()
    model.train()

    total_loss = 0.0
    steps = 0
    n_correct = 0
    n_possible = 0

    # Run batch gradient update
    for batch_idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda().double(), target.cuda().double()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        predicted_list, y_list = execute_sequence(data, target, model)

        pred = predicted_list[-1]
        y_ = y_list[-1].long()
        prediction = pred.data.max(1, keepdim=True)[1].long()
        n_correct += prediction.eq(y_.data.view_as(prediction)).sum().cpu().numpy()
        n_possible += int(prediction.shape[0])
        loss = F.nll_loss(pred, y_)

        loss.backward()
        optimizer.step()
        steps += 1
        total_loss += loss.cpu().data.numpy()
        optimizer.zero_grad()

    loss = total_loss / steps
    acc = (n_correct / n_possible)
    print("Train loss ", total_loss / steps)
    print("Train Accuracy ", (n_correct / n_possible))
    return loss, acc


def validate(epoch, model):
    dset.val()
    model.eval()

    total_loss = 0.0
    n_correct = 0
    n_possible = 0
    steps = 0

    # Run batch inference
    for batch_idx, (data, target) in enumerate(data_loader):

        if args.cuda:
            data, target = data.cuda().double(), target.cuda().double()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            predicted_list, y_list = execute_sequence(data, target, model)

        pred = predicted_list[-1]
        y_ = y_list[-1].long()
        prediction = pred.data.max(1, keepdim=True)[1].long()  # Index of the max log-probability
        n_correct += prediction.eq(y_.data.view_as(prediction)).sum().cpu().numpy()
        n_possible += int(prediction.shape[0])
        loss = F.nll_loss(pred, y_)

        steps += 1
        total_loss += loss.cpu().data.numpy()

    loss = total_loss / steps
    acc = n_correct / n_possible
    print("Validation Accuracy ", n_correct / n_possible)
    return loss, acc


# General training script logic
def run():
    model = create_model()
    model.double()
    params = 0
    for param in list(model.parameters()):
        params += param.numel()
    print("Num params: ", params)
    print(model)
    if args.cuda:
        model.cuda()
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join("runs", args.model_type + "_" + args.task, current_time + "_" + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir, comment="LR_{}_BATCH_{}".format(args.lr, args.batch_size))
    # best_val_loss = np.inf
    for epoch in range(args.epochs):
        print("\n\n**********************************************************")
        tim = time.time()
        loss, accuracy = train(epoch, model, optimizer)
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)
        with torch.no_grad():
            val_loss, val_accuracy = validate(epoch)
        writer.add_scalar("Loss/valid", val_loss, epoch)
        writer.add_scalar("Accuracy/valid", val_accuracy, epoch)
        print("Val Loss (epoch", epoch, "): ", val_loss)
        print("Epoch time: ", time.time() - tim)


def run_all():
    model_names = ["rnn", "lstm", "gru", "rnn+", "irnn", "peephole", "ugrnn"]
    for model_name in model_names:
        args.model_type = model_name
        model = create_model()
        model.double()
        params = 0
        for param in list(model.parameters()):
            params += param.numel()
        print("Num params: ", params)
        print(model)
        if args.cuda:
            model.cuda()
        model.double()
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        args.model_type = model_name
        log_dir = os.path.join("runs", args.model_type + "_" + args.task, current_time + "_" + socket.gethostname())
        writer = SummaryWriter(log_dir=log_dir, comment="LR_{}_BATCH_{}".format(args.lr, args.batch_size))

        # best_val_loss = np.inf
        for epoch in range(args.epochs):
            print("\n\n**********************************************************")
            tim = time.time()
            loss, accuracy = train(epoch, model, optimizer)
            writer.add_scalar("Loss/train", loss, epoch)
            writer.add_scalar("Accuracy/train", accuracy, epoch)
            with torch.no_grad():
                val_loss, val_accuracy = validate(epoch, model)
            writer.add_scalar("Loss/valid", val_loss, epoch)
            writer.add_scalar("Accuracy/valid", val_accuracy, epoch)
            print("Validation Loss (epoch", epoch, "): ", val_loss)
            print("Epoch time: ", time.time() - tim)

        # from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        # event_acc = EventAccumulator('/path/to/summary/folder')
        # event_acc.Reload()
        # # Show all tags in the log file
        # print(event_acc.Tags())
        #
        # # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
        # w_times, step_nums, vals = zip(*event_acc.Scalars('Accuracy'))
        #
        # plt.style.use('ggplot')  # Change/Remove This If you Want
        #
        # fig, ax = plt.subplots(figsize=(8, 4))
        # ax.plot(trees_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train', linewidth=4.0)
        # ax.plot(trees_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv', linewidth=1.0)
        # ax.fill_between(trees_grid, test_acc.mean(axis=1) - test_acc.std(axis=1),
        #                 test_acc.mean(axis=1) + test_acc.std(axis=1), color='#888888', alpha=0.4)
        # ax.fill_between(trees_grid, test_acc.mean(axis=1) - 2 * test_acc.std(axis=1),
        #                 test_acc.mean(axis=1) + 2 * test_acc.std(axis=1), color='#888888', alpha=0.2)
        # ax.legend(loc='best')
        # ax.set_ylim([0.88, 1.02])
        # ax.set_ylabel("Accuracy")
        # ax.set_xlabel("N_estimators")


if __name__ == "__main__":
    # run()
    run_all()
