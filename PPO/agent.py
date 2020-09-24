from networks import ActorNetwork, CriticNetwork
from torch.distributions.categorical import Categorical
import torch
from memory import ReplayBuffer



class Agent():
    def __init__(self, alpha, beta, input_dims, fc1_dims, fc2_dims, n_actions, eps_clip,batch_size):
        super(Agent,self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.eps_clip = eps_clip
        self.batch_size = batch_size
        self.max_grad_norm = 0.5
        self.actor = ActorNetwork(alpha=alpha, input_dims=input_dims, fc1_dims=fc1_dims, fc2_dims=fc2_dims,
                    n_actions=n_actions, name='actor')

                    
        self.critic = CriticNetwork(beta=beta, input_dims = input_dims, fc1_dims=fc1_dims, fc2_dims=fc2_dims,
                    n_actions=n_actions, name='critic')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.memory = ReplayBuffer(max_memory=3000, input_dims=self.input_dims, n_action=self.n_actions)
        
        self.gamma = .99
        self.trainning_step = 0

    
     


    def calculate_GAE(self,states_batch,rewards_batch,done_batch,gamma):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards_batch), reversed(done_batch)):
            if is_terminal:
                discounted_reward = 0
        
            discounted_reward = reward + (gamma * discounted_reward)
            rewards.insert(0,discounted_reward)
        
    
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        return rewards



    def actor_chose_action(self, state,action = None):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.actor.forward(state)
        probs = Categorical(action_probs)
     
        if action is None:
            action = probs.sample()
        
        
        
        
       
        return action.item() , action_probs[:, action.item()].item()
        

    def critic_get_value(self, state):
        state = torch.Tensor([state]).to(self.device)
        with torch.no_grad():
            val = self.critic.forward(state)
        return val.item()


    def learn(self):
        
        states_batch, actions_batch, rewards_batch, done_batch, log_probs_batch = self.memory.return_buffer()
        log_probs_batch = torch.tensor([log_probs_batch]).view(len(log_probs_batch), 1)
        GAE = self.calculate_GAE(states_batch, rewards_batch, done_batch, self.gamma)
        GAE = GAE.view(self.memory.mem_cntr,1)
        # print('GAE',GAE)
        state_feed = torch.tensor([states_batch]).view(self.memory.mem_cntr, self.input_dims)
        action_feed = torch.tensor([actions_batch]).view(self.memory.mem_cntr, 1)
        
        


        # print('state feed',state_feed)
        # print('action feed',action_feed)
        
        value = self.critic.forward(state_feed)
        # print('value', value)
        # input()
        advantages = (GAE - value).detach()

        action_policy = self.actor(state_feed).gather(1,action_feed)


   
        
        ratio = action_policy / log_probs_batch

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio,1-self.eps_clip,1+self.eps_clip)

        actor_loss = -torch.min(surr1, surr2).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor.optimizer.step()
        # print('value', value)
        # print('GAE',GAE)
        # input()

        value_loss = torch.nn.functional.mse_loss(GAE, value)
        self.critic.optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic.optimizer.step()


        self.memory.clear_buffer()
        self.trainning_step = self.trainning_step + 1
        






    
     
