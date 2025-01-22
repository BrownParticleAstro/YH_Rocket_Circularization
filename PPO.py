import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class RolloutBuffer : 
    
    def __init__(self):
        """
        Initiating the buffer tracking the following variables 
            - actions : action chosen by the agent (Actor) at each timestep
            - states : 4 dimensional state at each timestep
            - logprobs : log probability of the action taken 
            - state_values : Value of the state predicted by the critic
            - done : The episode ended 
            - max_step_done : The episode ended because it reached the maximum step 
    """
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.ep_done = []
        self.max_step_done = []

    def clear(self):
        """
        Clearing the buffer
    """
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.ep_done[:]
        del self.max_step_done[:]

########################## ACTOR CRITIC MODEL ##########################
class ActorCritic(nn.Module): 
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        
        # The Actor Neural Network 
        self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.LeakyReLU(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        # The Crtici Neural Network
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
    
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def act(self, state): 
        """"
        This function is used during the first loop of the training, when the actor solely interacts with the environment. 
        It is also the function used once the model has been trained and is just tested. 
    """
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat) # the model is using a normal distribution for the probability associated
        action = dist.sample()
        action_logprob = dist.log_prob(action) # The probability associated with the action chosen
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action): 
        """
        evaluate is used in the updating loop of the training, when new data is gathered to optimize the agent. 
        """
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy() # is revealing how much exploration the agent is taking which ensures maintaing a balance between exploration and exploitation
        state_values = self.critic(state) 

        return action_logprobs, state_values, dist_entropy

##################### Proximal Policy Optimization ##############################

class PPO_v1 : 
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init=0.6):
        
        self.action_std = action_std_init
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim, action_dim,action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ]) #specific learning rate for the actor and for the critic

        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to(device) #I'm wondering if this line is actually necessary
        self.policy_old.load_state_dict(self.policy.state_dict()) # apparently that copies the weights of the policy dict so that it starts at the same point, before any update is being made.

        self.MseLoss = nn.MSELoss() # Loss used for the critic 
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        '''
            The control of the standard deviation of the action probability is modified as follow 
                - It starts on a larger side, encouraging the model to explore 
                - It slowly decays until arriving to its minimum value in order to go from encouraging to explore to emphasizing the exploitation. 
        '''
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)

    def select_action(self, state) : 
        with torch.no_grad(): 
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.detach().cpu().numpy().flatten()

    def update(self): 
        """ 
        The update function can be seen as the second main loop used while training the model. 
        It is called when update_timestep and 
        """

        ### Creating a Discounted Reward List ### 
        max_step_reached = False 
        base_reward = 0 
        rewards = []
        discounted_reward = 0 

        for reward, max_step_done, ep_done in zip(reversed(self.buffer.rewards),reversed(self.buffer.max_step_done),reversed(self.buffer.ep_done)) : 
            if ep_done: 
                if max_step_done : 
                    discounted_reward = reward * (self.gamma**800 - 1)/(self.gamma -1)
                    base_reward = reward
                else : 
                    base_reward = 0 
                    discounted_reward = 0
                
            
            discounted_reward = reward + (self.gamma* discounted_reward) - base_reward*self.gamma**800
            rewards.insert(0,discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        ### Saving the list of old state and actions ####
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        ### Computing the Advantage ### 
        advantages = rewards.detach() - old_state_values.detach()

        ### Entering the updating loop and optimizing the policy and critic network ### 
        # We are upating it for K_epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions) # I'm confused because I don't really see where we precviously update the policy ...

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
            torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
            self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
            self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
