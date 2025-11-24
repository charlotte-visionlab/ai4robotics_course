import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sklearn.kernel_approximation import RBFSampler
# from sklearn.externals import joblib
import joblib
import sklearn.pipeline
import sklearn.preprocessing

# Policy representation adapted from
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html.
# The main takeaway here is not the structure itself as that is largely rule
# of thum and trial and error. But that this function approximation fits into
# the same algorithm as the linear case.
class MLPPolicy(nn.Module):

    def __init__(self):
        super(MLPPolicy, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head_a = nn.Linear(448, 2)
        self.head_v = nn.Linear(448, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        v = self.head_v(x.view(x.size(0), -1))
        x = self.head_a(x.view(x.size(0), -1))
        return x, v


class LinearPolicy(nn.Module):
    """ A feed forward policy representation """

    def __init__(self, env):
        super(LinearPolicy, self).__init__()
        n_inputs = env.observation_space.shape[0]
        n_outputs = env.action_space.n

        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])

        # Used to converte a state to a featurizes represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        self.featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf0", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf1", RBFSampler(gamma=2.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf3", RBFSampler(gamma=0.5, n_components=100)),
            ("rbf4", RBFSampler(gamma=0.25, n_components=100)),
        ])
        self.featurizer.fit(observation_examples)

        # policy
        self.fc1 = nn.Linear(500, n_outputs)

        # value approx
        self.fc1v = nn.Linear(500, 1)


    def save_feature_extractor(self, algo, e):
        joblib.dump(self.featurizer, 'saved_models/state_vec/sklearn_rbf_sampler_'+algo+'_'+str(e)+'.pkl')

    def load_feature_extractor(self, algo, e):
        self.featurizer = joblib.load('saved_models/state_vec/sklearn_rbf_sampler_'+algo+'_'+str(e)+'.pkl')

    def get_features(self, s):
        """
        Returns the featurized representation for a state.
        """
        featurized = self.featurizer.transform([s]) #scaled)
        return featurized[0]

    def forward(self, state):
        state = state.view(1, -1)
        x = self.fc1(state)
        v = self.fc1v(state)
        return x, v
