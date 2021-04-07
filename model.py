import torch
import torch.nn as nn
import torch.nn.functional as F

# We will use a neural network with 3 fully connected layers to define our policy.

class ActorNet(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
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
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.std = nn.Parameter(torch.zeros(action_size))
        self.actor_params = list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(self.fc3.parameters())
        self.actor_params.append(self.std)

    # forward step: feed forward the input (state) through 3 FC layers and 2 ReLUs to get the estimated action valus. 
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        out_values = torch.tanh(self.fc3(x))
#        out_values.backward()
        
 #       print("out_values:", out_values)
 #       print("std: ", self.std)
 #       print("sstd: ", F.softplus(self.std))
        dist = torch.distributions.Normal(out_values, F.softplus(self.std))
        action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
 #       print("action:", action)
        #print("entropy", entropy)
        
        """
        probs = policy_network(state)
        #Note that this is equivalent to what used to be called multinomial
        m = Categorical(probs)
        action = m.sample()
        next_state, reward = env.step(action)
        loss = -m.log_prob(action) * reward
        loss.backward()
        """
        return ( action.squeeze(0).cpu().detach().numpy(), log_prob, entropy )

class CriticNet(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
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
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.fc3(x)
        #return v.squeeze(0).cpu().detach().numpy()
        return v.squeeze()   #.cpu().detach().numpy()
