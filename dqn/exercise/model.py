import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear( state_size, 256 )
        self.fc2 = nn.Linear( 256, 128 )
        self.fc3 = nn.Linear( 128, 64 )
        self.fc4 = nn.Linear( 64, action_size )
        

    def forward(self, state):
        
        x = F.relu( self.fc1( state ) )
        x = F.relu( self.fc2( x ) )
        x = F.relu( self.fc3( x ) )
        x = self.fc4( x )
        
        return x

