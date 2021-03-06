import numpy as np
import random
from model import ActorNet, CriticNet
import torch
import torch.nn.functional as F
import torch.optim as optim

"""
The "A2C_Agent" class implements the agent learning to solve the Continuous Control environment problem using an Actor-Critic method (see report.md for more info).  
"""

class A2C_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, device, seed, LR=5e-4, gamma=0.95, entropy_weight=0.02, actor_network_max_grad_norm = 5, critic_network_max_grad_norm = 5, nstepqlearning_size=5, gae_lambda = 1.0):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            LR (float): learning rate
            GAMMA (float): factor used to discount future values
            entropy_weight (float): weight of the entropy value using with the entropy_loss
            actor_network_max_grad_norm (float): threshold value used in gradient clipping in the actor model 
            critic_network_max_grad_norm (float): threshold value used in gradient clipping in the critic model 
            nstepqlearning_size (int): the number of steps used for the N-step bootstrapping algorithm
            gae_lambda (float): lambda used in GAE algorithm to use as discount factor in getting a mixture of every available estimated N-step bootstrapping results (from 1-5). 

        """
        self.state_size = state_size
        self.action_size = action_size
        self.entropy_weight = entropy_weight
        random.seed(seed)
        self.gamma=gamma
        self.actor_network_max_grad_norm = actor_network_max_grad_norm
        self.critic_network_max_grad_norm = critic_network_max_grad_norm
        self.nstepqlearning_size = nstepqlearning_size
        self.gae_lambda = gae_lambda
        self.device=device

        print("----Dumping agent hyperparameters---- ")
        print("LR: ", LR)
        print("gamma: ", gamma)
        print("actor_network_max_grad_norm: ", self.actor_network_max_grad_norm)
        print("critic_network_max_grad_norm: ", self.critic_network_max_grad_norm)
        print("nstepqlearning_size: ", self.nstepqlearning_size)
        print("gae_lambda: ", self.gae_lambda)
        print("entropy_weight: ", self.entropy_weight)
        print("------------------------------------- ")

        self.actor_net = ActorNet(state_size, action_size, device, seed).to(self.device)       # Theta
        self.critic_net = CriticNet(state_size, action_size, seed).to(self.device)     # Thetav
        self.actor_optimizer = optim.RMSprop(self.actor_net.parameters(), lr=LR)
        self.critic_optimizer = optim.RMSprop(self.critic_net.parameters(), lr=LR)
    
    def tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x
        x = np.asarray(x, dtype=np.float32)
        x = torch.from_numpy(x).to(self.device)
        return x

    def act(self, state):
        """Returns actions for given state as per current policy.
        Params
        ======
            state (array_like): current state
        """

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.actor_net.eval()
        self.critic_net.eval()
        ( actor_values, log_prob, entropy ) = self.actor_net(state)
        critic_values = self.critic_net(state)
        self.actor_net.train()
        self.critic_net.train()
        return actor_values, log_prob, entropy, critic_values

    def train_one_episode(self, env, brain_name):
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
        num_agents = len(env_info.agents)

        states = env_info.vector_observations                  # get the current state (for each agent)
        episode_terminated = False
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)

        while episode_terminated == False:
            l_states = []
            l_actions = []
            l_rewards = []           #np.zeros(( nstepqlearning_size, num_agents ))
            l_masks = []
            l_next_states = []
            l_values = []
            l_log_probs = []
            l_entropy = []

            nstep_memory_size = self.nstepqlearning_size 
            for i in range(self.nstepqlearning_size):

                # Get a(t) according to actor policy 
                (actions, log_prob, entropy, values) = self.act(states)
                actions = np.clip(actions, -1, 1)                  # Put all actions between -1 and 1. ( The last activation of the Actor is tanh, which puts the out values in this range, 
                                                                   # but later we are sampling it which can produce values outside of this range)
                
                # Perform a(t) in all environments
                env_info = env.step(actions)[brain_name]           # send all actions to tne environment

                # get s(t+1), r(t) and wasLastAction(t)
                next_states = env_info.vector_observations         # get next state (for each agent)
                rewards = env_info.rewards                         # get reward (for each agent)
                dones = env_info.local_done                        # see if episode finished

                masks = 1 - np.asarray(dones, np.int)

                l_states.append(states)
                l_actions.append(actions)
                l_rewards.append( rewards )
                l_masks.append(masks)
                l_next_states.append(next_states)
                l_values.append( values )
                l_log_probs.append( log_prob )
                l_entropy.append(entropy)

                # update score
                scores += env_info.rewards                         # update the score (for each agent)

                states = next_states                               # roll over states to next time step
                if np.any(dones):                                  # exit loop if episode terminated
                    nstep_memory_size = i + 1
                    episode_terminated = True
                    break

            # get one prediction for the last estimated Q value
            (_, _, _, values) = self.act(states)
            l_values.append( values )    # Add to the list, GAE will use it
            
            advantages = self.tensor( torch.zeros((num_agents)) ).to(self.device)
            returns = values.reshape(( num_agents, )).to(self.device)     # last estimated Value ( of s(t+nstep_memory_size) )

            l_advantages = [None] * nstep_memory_size   
            l_rets = [None] * nstep_memory_size          
            l_masks = torch.tensor(np.array(l_masks)).to(self.device)        
            l_rewards = torch.tensor(np.array(l_rewards)).to(self.device)    

            for i in reversed(range(nstep_memory_size)):
                returns = l_rewards[i] + self.gamma * l_masks[i] * returns
                
                # Normal advantage calculation. 
                #advantages = returns - l_values[i].detach().reshape((num_agents, ))
                
                # GAE
                td_error = l_rewards[i] + self.gamma * l_masks[i] * l_values[i+1] - l_values[i]
                advantages = advantages * self.gae_lambda * self.gamma * l_masks[i] + td_error
                # GAE end 
                
                l_advantages[i] = advantages.detach() 
                l_rets[i] = returns.detach()

            # bring log_probs list to Tensor with shape [ num_agents,nstepqlearning_size ] 
            logprobs = torch.cat(l_log_probs).squeeze()            
            logprobs = logprobs.reshape(( nstep_memory_size*num_agents )).to(self.device)     

            ents = torch.cat(l_entropy).squeeze()
            advantages_tensor = torch.cat(l_advantages, dim=0).squeeze().detach().to(self.device)

            policy_loss = -(logprobs * advantages_tensor).mean()
            
            # entropy: currently it's constant but I left it here, to make it possible to use different distribution parameters during the training process
            entropy_loss = ents.mean()

            # ==== train Critic ====
            self.critic_optimizer.zero_grad()
            l_rets = torch.cat(l_rets, dim=0).squeeze().detach().to(self.device)
            l_values = torch.cat(l_values[:nstep_memory_size], dim=0).squeeze().to(self.device)
            v = 0.5 * ( l_rets - l_values )
            value_loss = v.pow(2).mean() 
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.critic_network_max_grad_norm)
            self.critic_optimizer.step()

            # ==== train Actor ====
            self.actor_optimizer.zero_grad()
            # Add entropy term to the loss function to encourage having evenly distributed actions 
            (policy_loss - self.entropy_weight * entropy_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.actor_network_max_grad_norm)
            self.actor_optimizer.step()


        return scores


