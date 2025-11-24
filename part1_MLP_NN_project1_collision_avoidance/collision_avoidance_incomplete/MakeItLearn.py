import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PreProcessing import PreprocessData


# Set Seeds For Randomness
torch.manual_seed(10)
np.random.seed(10)
InputSize = 6  # Input Size
batch_size = 1  # Batch Size Of Neural Network
NumClasses = 1  # Output Size

############################################# FOR STUDENTS #####################################

NumEpochs = 25
HiddenSize = 10


# Create The Neural Network Model
class Net(nn.Module):
    def __init__(self, InputSize, NumClasses):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(InputSize, HiddenSize)
        self.fc2 = nn.Linear(HiddenSize, NumClasses)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


net = Net(InputSize, NumClasses)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)

##################################################################################################

if __name__ == "__main__":

    TrainSize, SensorNNData, SensorNNLabels = PreprocessData()
    for j in range(NumEpochs):
        losses = 0
        for i in range(TrainSize):
            input_values = Variable(SensorNNData[i])
            labels = Variable(SensorNNLabels[i])
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = net(input_values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses += loss.item()

        print('Epoch %d, Loss: %.4f' % (j + 1, losses / SensorNNData.shape[0]))
        torch.save(net.state_dict(), './SavedNets/NNBot.pkl')
