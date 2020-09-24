import os
import torch as T
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal







class ActorNetwork(nn.Module):
    def __init__(self, alpha,input_dims,fc1_dims, fc2_dims,n_actions ,name, chkpt_dir='/home/samiul/Documents/RELN/PPO_self/tmp/ppo_self'):
        super(ActorNetwork, self).__init__()
    
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
     
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
    


        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.action_probs = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        prob = self.action_probs(prob)
        prob = F.softmax(prob,dim =1)
   
        return prob
        

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file,map_location=T.device('cpu')))


        
        
        
    
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims,fc1_dims,fc2_dims,n_actions,name, chkpt_dir='/home/samiul/Documents/RELN/PPO_self/tmp/ppo_self'):
        super(CriticNetwork, self).__init__()
   
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.input_dims = input_dims

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ppo')

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        
        self.q1 = nn.Linear(self.fc1_dims, 1)
    


        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)




    def forward(self, state):
        q1_action_value = self.fc1(state)
        q1_action_value = F.relu(q1_action_value)
        q_value = self.q1(q1_action_value)
    
      
        return q_value


    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file,map_location=T.device('cpu')))



