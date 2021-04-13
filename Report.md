

[TOC]



# Project 2: Continuous Control - Report

This project was about making and train an RL agent to solve the Reacher environment.  

I have chosen Advantage Actor Critic (A2C) with Generalized Advantage Generalization (GAE) and N-step bootstrapping methods for the agent because: 

- I wanted to implement an Actor Critic method. For me, it seems to be the state of the art (or at least very new) and challenging algorithm in Reinforcement Learning. 

- A2C produces comparable performance to A3C, but sometimes it's more efficient and the Udacity environment used is synchronous by design. 

  

 

## Architecture

I have create a hybrid solution in python using the numpy and pytorch frameworks for the agent, and it is using the UnityEnvironment module from the unityagents library for establishing the bridge between the agent and the simulator (written in Unity). 

The agent can be trained using a Jupyter notebook. The files:

- model.py: The neural networks implementing the policy model (the Actor) and the value approximator model (the Critic).

- p2_agent.py: Here is implemented the A2C agent: The "A2C_Agent " class implements the A2C algorithm. Only its constructor, "act", and "train_one_episode" methods are called from outside.

- Navigation.ipynb: A Jupyter notebook for demonstrating 

  - the train process of the agent, 
  - the visualized results (scores),
  - after training, the model is loaded and an episode is played as a demonstration.   

  This file is also exported here, so you can read it without an installed Jupyter notebook instance: [exported notebook](export/Continuous_Control.md)



Please follow the notebook to see the training of the Continuous_Control agent. 

For the detailed description of the algorithm used in the agent and the models, you can use the comments in the relevant python codes, and you can also use the following text.    



## The implemented A2C RL Agent

### Design of the model 

The agent is using 2 different neural networks:

- The Actor model (ActorNet class) is the policy network which decides the actions the agent. It is using 3 Fully Connected layers with RELU activations between them, and a tanh layer at the end. The layer sizes are [64,64,4]. The hyperbolic tangent function normalizes the 4 output values in the range (-1,1).  After that we are sampling actions from these outputs using a normal (Gaussian) distribution with the values as mean and with a predefined sigma standard deviation (hyperparameter). This last step is necessary to implement stochasticity. So, from the Actor network we get the following data: 
  - the sampled actions,
  - the log of the probabilities of the sampled actions, 
  - the entropy of the distribution (constant, because the standard deviation is a hyperparameter)
- The second network is the Critic (CriticNet class). It's task is to approximate the Value of a given state. It is also using 3 Fully Connected layers and 2 RELU activation function layers between them. The layer sizes are [64,64,1]. The final output is not modified, it is the real, estimated Value of the input state.   

The A2C_Agent is using these two networks to implement the training, and using them to replay episodes. 

### Training process

#### Main loop

The training is happening by playing full episodes with 20 simulated environments parallelly, but in a synchronized way (all 20 agent actions go to, and all 20 rewards & next states come from the simulator once per timestep).

The average of the 20 total scores of the episodes is recorded. When these scores reach 30.0 in the last 100 episodes we consider the problem solved. ( solved in the episode after which this moving average in the next 100 episodes is above 30.0 )

Anyway, we continue the training until the average score reaches 35.0 and save this very professional agent model :)

#### Agent initialization

The final agent is initialized with the following (tuned) hyperparameters:

LR = 0.0007 					   # Learning Rate used by the optimizer.

nstepqlearning_size = 5 	# the number of steps used for the N-step bootstrapping algorithm

gamma = 0.925 				 # gamma used to discount future rewards.  

gae_lambda = 1.0 			# lambda used in GAE algorithm to use as discount factor in getting a mixture of every available estimated N-step bootstrapping results (from 1-5). 

actor_network_max_grad_norm = 15		#  Max. norm of the gradients in the Actor network to avoid exploding gradient problem 

critic_network_max_grad_norm = 15		#  Max. norm of the gradients in the Critic network to avoid exploding gradient problem 

sigma = F.softplus( 0 )                             # 0.6931. It is used as scale value in the Normal distribution.

#### Playing one episode (and train on it)

To play an episode and train the models with it can be achieved by calling the agent's "train_one_episode" method (this is happening in the earlier described Main loop).

The basic workflow if the following:

- First, we collect data from the next "nstepqlearning_size" steps, and the estimated Value of the last step (this last one is required by the GAE algorithm). 
- Next, we calculate the advantages and the estimated discounted Value for each of these steps
- Then we calculate the following loss values to train the models:
  - policy_loss: This is the policy loss which uses the log probability of the sampled actions, and the advantages calculated for the next [nstepqlearning_size] steps. It will be used to train the Actor network.
  - entropy_loss: This is the entropy of the distribution used to get the finally chosen actions. It will be used to train the Actor network. During hyperparameter tuning I've found that the scale of the Normal distribution (from which the real action is sampled) is better to be a constant value. It also means that this entropy_loss will also be a constant in the final agent. Anyway, I did not remove it from the final algorithm, as all the tunings of the hyperparameters considered this constant loss's presence.      
  - value_loss: Loss value, it's based on the difference between the total estimated returns using the Critic-estimated Value of last state  s(t+nstepqlearning_size ) plus the discounted rewards in each timesteps in the N-step buffer, and the Critic-estimate of all recorded states from s(t)  to s(t+nstepqlearning_size-1) . It will be used to train the Critic network.
- As the last step we train the Critic and the Actor networks using the RMSProp optimizer. We then continue by executing this loop from the t+nstepqlearning_size timestep if none of the 20 agents terminated earlier.

### Model optimization

At first, I have used the regular Adam optimizers for both the Actor and Critic networks. Unfortunately, the learning process stopped on a level of 0.8-1.2 average score and did not improve further. Maybe it could not continue after reaching a local maximum. Even with different Learning Rates the resulting networks did not improve after a time.

The authors of the referenced A3C paper also investigated three different optimizers and finally they concluded that the RMSProp is the best. So, I also gave it a try and found out that it's really much better, the agent improved continuously after finding the best learning rate value.          

Moreover, while optimizing the hyperparameters I have found: 

- With bigger gamma values (for ex.: 0.99) the scores will not improve, or will improve very slowly. Discounting future rewards/values more aggressively is better.  
- Lowering the Learning Rate causes steadier/more balanced learning, but the cost is that it requires more episodes to solve the task. 

### Results, Conclusion

During hyperparameter tuning my experience: Actor Critic Reinforcement Learning methods can be very effective in learning if the hyperparameters are good. Unfortunately, wrong parameters often cause complete failure in learning.       

It can be seen in the notebook, the environment was solved in 41 episodes, and reached score 35.0 in just 59 episodes (average score reached 35.0 in episode 60-139) ! 

At the final part of the jupyter notebook, after reaching score 35.0, I've demonstrated: 

- how to save the final model, 

- then I've shown how to load the networks after reinitialization,

- and how to (just) play an episode, without training. 

- Moreover, in the last step I've visualized the distribution of the results among the participating agents. 

  

## Ideas for future work

After this implementation, I think that the following ideas and promising directions would be worthwhile to work on: 

- I've used a constant standard deviation value in the Normal distribution when I sampled the actions. It would worth to try using some continuously decaying value here.

- Hyperparameter tuning: There's never enough CPU/GPU time. I've executed many trainings, but it's very slow so I could not investigate all parameter combinations.  

- It would be interesting to increase/decrease the number of agents in the Unity simulator and check the effect of it on the while training time.

   

## REFERENCES

[Volodymyr Mnih](https://arxiv.org/search/cs?searchtype=author&query=Mnih%2C+V), [Adrià Puigdomènech Badia](https://arxiv.org/search/cs?searchtype=author&query=Badia%2C+A+P), [Mehdi Mirza](https://arxiv.org/search/cs?searchtype=author&query=Mirza%2C+M), [Alex Graves](https://arxiv.org/search/cs?searchtype=author&query=Graves%2C+A), [Timothy P. Lillicrap](https://arxiv.org/search/cs?searchtype=author&query=Lillicrap%2C+T+P), [Tim Harley](https://arxiv.org/search/cs?searchtype=author&query=Harley%2C+T), [David Silver](https://arxiv.org/search/cs?searchtype=author&query=Silver%2C+D), [Koray Kavukcuoglu](https://arxiv.org/search/cs?searchtype=author&query=Kavukcuoglu%2C+K) - *Asynchronous Methods for Deep Reinforcement Learning*  -  https://arxiv.org/abs/1602.01783

[John Schulman](https://arxiv.org/search/cs?searchtype=author&query=Schulman%2C+J), [Philipp Moritz](https://arxiv.org/search/cs?searchtype=author&query=Moritz%2C+P), [Sergey Levine](https://arxiv.org/search/cs?searchtype=author&query=Levine%2C+S), [Michael Jordan](https://arxiv.org/search/cs?searchtype=author&query=Jordan%2C+M), [Pieter Abbeel](https://arxiv.org/search/cs?searchtype=author&query=Abbeel%2C+P) - *High-Dimensional Continuous Control Using Generalized Advantage Estimation*  - https://arxiv.org/abs/1506.02438 









