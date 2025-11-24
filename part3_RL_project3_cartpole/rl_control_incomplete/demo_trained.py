
import argparse
import sys
import os
from PIL import Image
import torch.nn.functional as F

import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as T

import numpy as np

import models
# import policy_grad
import policy_grad_solution
from simulator.cartpole import CartPoleEnv
# Composition of functions to take tensor and convert to network input.

MODEL_IS_RBF = False

screen_width = 600

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.BICUBIC),
                    T.ToTensor()])

def get_cart_location():
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)
    # Strip off the top and bottom of the screen
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = get_cart_location()
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)

def eval(env, model, model_is_rbf=False, avgn=5, render=False):
    eval_rewards = []
    model.eval()
    if render:
        env.reset()
        env.render()

    eval_reward = 0.0
    for i in range(avgn):
        done = False
        obs = env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen

        while not done:
            if render:
                env.render()

            last_screen = current_screen
            current_screen = get_screen()
            state = current_screen - last_screen

            if model_is_rbf:
                features = model.get_features(obs)
                state = Variable(torch.from_numpy(features)).float().to(device)

            action_lin, value = model(state)

            action_probs = F.softmax(action_lin, dim=1)
            action = torch.max(action_probs, dim=1)[1]
            action_cpu = action.cpu()
            obs, reward, done, _ = env.step(action_cpu.data.numpy().squeeze())
            eval_reward += reward

    eval_reward /= avgn
    eval_rewards.append(eval_reward)
    sys.stdout.write("\r\nEval reward: %d \r\n" % (eval_reward))
    sys.stdout.flush()

    return eval_reward

if __name__ == '__main__':
    env = CartPoleEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_model_pixels = 'saved_models/state_pixels/model_aac_70.pth'
    eval_model_rbf = 'saved_models/state_vec/model_aac_190.pth'
    if MODEL_IS_RBF:
        eval_model = eval_model_rbf
    else:
        eval_model = eval_model_pixels

    model = torch.load(eval_model, weights_only = False)
    filename = os.path.basename(eval_model)
    print(f'Loaded model {filename} for evaluation')
    eval(env, model, model_is_rbf=MODEL_IS_RBF, render=True)

