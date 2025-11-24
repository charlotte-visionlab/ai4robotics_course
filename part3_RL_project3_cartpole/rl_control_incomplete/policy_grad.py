import torch
from torch.autograd import Variable

def compute_losses(R, ep_rewards, ep_values, ep_action_log_probs, masks, gamma, tau, algo):
    """ Compute the policy and value func losses given the terminal reward, the episode rewards,
    values, and action log probabilities """
    policy_loss = 0.0
    value_loss = 0.0

    # TODO: Use the available parameters to compute the policy gradient loss and the value function
    # loss.
    raise NotImplementedError("Compute the policy and value function loss.")
    return policy_loss, value_loss
