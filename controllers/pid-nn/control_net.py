import torch
import torch.nn as nn

from constants.controller_constants import *

class ControlNet(nn.Module):
    def __init__(self):
        super(ControlNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(CONTROLLER_INPUT_DIM, CONTROLLER_HIDDEN_DIM1, bias=True),
            nn.ReLU(),
            nn.Linear(CONTROLLER_HIDDEN_DIM1, CONTROLLER_HIDDEN_DIM2, bias=True),
            nn.ReLU(),
            nn.Linear(CONTROLLER_HIDDEN_DIM2, CONTROLLER_OUTPUT_DIM, bias=True)
        )

    def forward(self, x):
        return self.layers(x)
