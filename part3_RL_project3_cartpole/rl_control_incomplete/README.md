# Implementation of Policy Gradients applied to the Cart Pole problem

See requirements.txt for a list of required python packages. This code was
written for python3.6 and was not tested on python2.x.

Simulator code is located in the simulator directory. Data is saved to logs, saved_models.

## Learning from a vector states

The script main_state_vec.py implements a basic policy gradient algorithm with
a linear (RBF) function approximator. Since there is a fixed point where the
pole is vertical this can be learned with a linear model.

`python main_state_vec.py` will train the model and save models and logs to
saved_models/state_vec/ and logs/state_vec.

To view a trained model run
`python main_state_vec --eval-model models/state_vec/some_model.pth`

## Learning from a pixels.

This script implements the same algorithm but does so learning directly from
pixel input. The training and evaluation functionality is the same as with
the previous script.
