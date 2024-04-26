import torch

class MyNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)