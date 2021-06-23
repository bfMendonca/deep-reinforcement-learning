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
        
        self.fc1 = nn.Linear( state_size, 1024 )
        self.fc2 = nn.Linear( 1024, 512 )
        self.fc3 = nn.Linear( 512, action_size )
        

    def forward(self, state):
        
        x = F.relu( self.fc1( state ) )
        x = F.relu( self.fc2( x ) )
        x = self.fc3( x )
        
        return x


    
class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        #Advantage Stream
        self.afc1 = nn.Linear( state_size, 1024 )
        self.afc2 = nn.Linear( 1024, 512 )
        self.afc3 = nn.Linear( 512, action_size )

        #State-Value Strean
        self.vfc1 = nn.Linear( state_size, 512 )
        self.vfc2 = nn.Linear( 128, 64 )
        self.vfc3 = nn.Linear( 64, action_size )
        

    def forward(self, state):
        
        adv = F.relu( self.afc1( state ) )
        adv = F.relu( self.afc2( adv ) )
        adv = self.afc3( adv )

        val = F.relu( self.vfc1( state ) )
        val = F.relu( self.vfc2( val ) )
        val = self.vfc3( val )

        out = val + (adv - adv.mean() )
        
        return out

