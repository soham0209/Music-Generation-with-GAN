#!/usr/bin/env python

# Generative Adversarial Networks (GAN) example in PyTorch.
# See related blog post at
# Source: https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f#.sch4xgsa9
# Modified by Soham Mukherjee,Pavani Komati and Rupendra Nath Mitra for CSE 5523: Final Project


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import glob
import data_utils_compose
import data_utils_train


chord_train_dir = './trainData/chords/'
mel_train_dir = './trainData/melody/'
composition_dir = './output/'

# Data params
chord_train_files = glob.glob("%s00*.mid" % chord_train_dir)
mel_train_files = glob.glob("%s00*.mid" % mel_train_dir)
print("Choose a resolution factor. "
      "(e.g. Resolution_Factor=24: 1/8 Resolution, 12: 1/16 Resolution, 6: 1/32 Resolution, etc...)")
resolution_factor = 12

# Preprocessing: Get highest and lowest notes + maximum midi_ticks overall midi files
chord_lowest_note, chord_highest_note, chord_ticks = data_utils_train.getNoteRangeAndTicks(chord_train_files,
                                                                                           res_factor=resolution_factor)
mel_lowest_note, mel_highest_note, mel_ticks = data_utils_train.getNoteRangeAndTicks(mel_train_files,
                                                                                     res_factor=resolution_factor)

# Create Piano Roll Representation of the MIDI files.
# Return: 3-dimensional array or shape (num_midi_files, maximum num of ticks, note range)

chord_roll = data_utils_train.fromMidiCreatePianoRoll(chord_train_files, chord_ticks, chord_lowest_note,
                                                      chord_highest_note, res_factor=resolution_factor)
mel_roll = data_utils_train.fromMidiCreatePianoRoll(mel_train_files, mel_ticks, mel_lowest_note, mel_highest_note, res_factor=resolution_factor)

# Double each chord_roll and mel_roll. Preprocessing to create Input and Target Vector for Network

double_chord_roll = data_utils_train.doubleRoll(chord_roll)
double_mel_roll = data_utils_train.doubleRoll(mel_roll)

# Create Network Inputs:
# Input_data Shape: (num of training samples, num of timesteps=sequence length, note range)
# Target_data Shape: (num of training samples, note range)

input_data, target_data = data_utils_train.createNetInputs(double_mel_roll, double_mel_roll, seq_length=mel_ticks)
input_data = input_data.astype(np.bool)
input_data = input_data.astype(np.uint8)
target_data = target_data.astype(np.bool)
target_data = target_data.astype(np.uint8)


input_dim = input_data.shape[2]
output_dim = input_data.shape[2]


print()
print()

# Model params
g_input_size = input_dim    # Random noise dimension coming into generator, per output vector
g_hidden_size = 50   # Generator complexity
g_output_size = output_dim    # size of generated output vector

d_input_size = input_dim   # Minibatch size - cardinality of distributions
d_hidden_size = 50   # Discriminator complexity
d_output_size = 1    # Single dimension for 'real' vs. 'fake'
# minibatch_size = d_input_size

d_learning_rate = 2e-4  # 2e-4
g_learning_rate = 2e-4
optim_betas = (0.9, 0.999)
print("For how many epochs do you wanna train?")
num_epochs = int(input('Num of Epochs:'))
print_interval = 1
d_steps = 1  # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator
g_steps = 1

(name, preprocess, d_input_func) = ("Raw data", lambda data: data, lambda x: x)

print("Using data [%s]" % name)


# def distribution_sample():
#     result = np.dstack((fft_data[0], fft_data[1])).flatten()
#     result2 = np.reshape(result, (len(raw_data), 4))
#     return torch.Tensor(result2)


def noise():
    # tf = np.array(n[.ra], dtype=np.uint8)
    return torch.Tensor(np.random.uniform(0, 1, input_data.shape))  # Uniform-dist data into generator, _NOT_ Gaussian


# ##### MODELS: Generator model and discriminator model

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = torch.sigmoid(self.map2(x))
        return self.map3(x)


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.elu(self.map2(x))
        return torch.sigmoid(self.map3(x))


G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
D = Discriminator(input_size=d_input_func(d_input_size), hidden_size=d_hidden_size, output_size=d_output_size)
criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)

# dataset = distribution_sample()
dataset = torch.Tensor(input_data)
g_fake_data = None

# For Plots
t = np.arange(0, num_epochs, 1, dtype=int)
G_error = list()
D_error = list()

for epoch in range(num_epochs):
    for d_index in range(d_steps):
        # 1. Train D on real+fake
        D.zero_grad()

        #  1A: Train D on real
        d_real_data = Variable(dataset)
        d_real_decision = D(d_real_data)
        d_real_error = criterion(d_real_decision, Variable(torch.ones(d_real_decision.shape)))  # ones = true
        d_real_error.backward()  # compute/store gradients, but don't change params

        #  1B: Train D on fake
        d_gen_input = Variable(noise())
        d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
        d_fake_decision = D(d_fake_data)
        d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(d_fake_decision.shape)))  # zeros = fake
        d_fake_error.backward()
        d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()
        D_error.append(d_fake_error.item())

    for g_index in range(g_steps):
        # 2. Train G on D's response (but DO NOT train D on these labels)
        G.zero_grad()

        gen_input = Variable(noise())
        g_fake_data = G(gen_input)
        dg_fake_decision = D(g_fake_data)
        g_error = criterion(dg_fake_decision, Variable(torch.ones(dg_fake_decision.shape)))  # we want to fool,
        G_error.append(g_error.item())
        g_error.backward()
        g_optimizer.step()  # Only optimizes G's parameters
        print(epoch, " : D: ", d_fake_error.item(), " G: ", g_error.item())

fake_data = g_fake_data.detach().numpy()
thresh = 0.3  # float(input('Threshold (recommended ~ 0.1):'))
net_roll = data_utils_compose.NetOutToPianoRoll(fake_data, threshold=thresh)
# print("net_roll:", net_roll)
# print("net_roll.shape", net_roll.shape)
data_utils_compose.createMidiFromPianoRoll(net_roll, mel_lowest_note, composition_dir,
                                           2, thresh, res_factor=resolution_factor)

print("Finished composing song %d." % 2)

plt.ioff()
fig = plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.xticks(range(0, num_epochs, 100))
plt.plot(range(0, num_epochs), D_error, 'g', ms=3, ls='-', lw=1,
         label='Discriminator Error on Fake Data')
plt.plot(range(0, num_epochs), G_error, 'r', ms=3, ls='--', lw=1,
         label='Generator Error')
plt.legend()
plt.savefig('Error_Details.png')
plt.close(fig)


