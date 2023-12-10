import torch
import torch.nn as nn
import torch.optim as optim


class ResidualBlock(nn.Module):
    def __init__(self, inp_size, n_fc, fc_size):
        super(ResidualBlock, self).__init__()
        layers = []
        layers.append(nn.Linear(inp_size, fc_size))
        layers.append(nn.ReLU())
        
        for _ in range(1, n_fc - 2):
            layers.append(nn.Linear(fc_size, fc_size))
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)
        self.output1 = nn.Linear(fc_size, inp_size)
        self.output2 = nn.Linear(fc_size, 1)

    def forward(self, x):
        x = self.layers(x)
        x_bc = self.output1(x)
        x_fc = self.output2(x)
        return x_bc, x_fc
    

class ResidualNeuralNet(nn.Module):
    def __init__(self, inp_size, n_fc, fc_size, n_blocks):
        super(ResidualNeuralNet, self).__init__()
        self.fc_size = fc_size
        self.blocks = nn.ModuleList([ResidualBlock(inp_size, n_fc, fc_size) for i in range(n_blocks)])

    def forward(self, x):
        x_bc, x_fc = self.blocks[0](x)
        cur_x = x - x_bc
        output = x_fc
        for i in range(len(self.blocks) - 1):
            x_bc, x_fc = self.blocks[i](cur_x)
            cur_x = cur_x - x_bc
            output = output + x_fc
        return output