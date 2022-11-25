import torch.nn as nn


class Yoga_Neural_Network(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Yoga_Neural_Network, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self.layer_3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        yoga_output = self.layer_1(x)
        yoga_output = self.relu(yoga_output)
        yoga_output = self.layer_2(yoga_output)
        yoga_output = self.relu(yoga_output)
        yoga_output = self.layer_3(yoga_output)

        return yoga_output
