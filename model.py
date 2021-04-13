import torch
import torch.nn as nn
import torch.nn.functional as F

# We will use a 2 neural networks each with 3 fully connected layers to define our policy and value approximator.

class ActorNet(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, device, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            device (torch.device): Device to execute the network on)
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """        
        super().__init__()
        print("Creating model with ", fc1_units, " and ", fc2_units, " hidden layer sizes." )
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
        # Let the standard deviation be constant (hyperparameter)
        self.sigma = F.softplus( torch.zeros(action_size).to(device) )
        
    # forward step: feed forward the input (state) through 3 FC layers and 2 ReLUs to get the estimated action valus. 
    def forward(self, state):
        """Build a network that maps state -> actions, log probabilitis and entropy of the distribution."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        out_values = torch.tanh(self.fc3(x))

        # create a normal distribution and sample the actions from it.
        dist = torch.distributions.Normal(out_values, self.sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        
        return ( action.squeeze(0).cpu().detach().numpy(), log_prob, entropy )

class CriticNet(nn.Module):
    """Critic (Value estimator) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """        
        super().__init__()
        print("Creating model with ", fc1_units, " and ", fc2_units, " hidden layer sizes." )
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

    # forward step: feed forward the input (state) through 3 FC layers and 2 ReLUs to get the estimated action valus. 
    def forward(self, state):
        """Build a network that maps state -> expected Q values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.fc3(x)
        return v.squeeze()   

