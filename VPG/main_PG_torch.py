#vanila policy gradiant with torch

import torch
import os
import  numpy as np
import gym
import matplotlib.pyplot as plt


class policyNetwork(torch.nn.Module):
    def __init__(self, policy_input_dims, policy_fc1_dims, policy_fc2_dims, policy_lr, policy_n_actions):
        super(policyNetwork, self).__init__()
        self.policy_input_dims=policy_input_dims
        self.policy_fc1_dims =policy_fc1_dims
    
        self.policy_fc2_dims=policy_fc2_dims
        self.policy_lr=policy_lr
        self.policy_n_actions=policy_n_actions

 
        

        self.fc1=torch.nn.Linear(*self.policy_input_dims, self.policy_fc1_dims)
        self.fc2=torch.nn.Linear(self.policy_fc1_dims, self.policy_fc2_dims)
        self.output =torch.nn.Linear(self.policy_fc2_dims, self.policy_n_actions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=policy_lr, weight_decay=.01)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self,states):
        x = self.fc1(states)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = self.output(x)

        action_dist = torch.nn.functional.softmax(x)

        return action_dist






class Agent():
    def __init__(self, input_dims, fc1_dims, fc2_dims, lr, n_actions,gamma):
        super(Agent, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.policy_function = policyNetwork(self.input_dims, self.fc1_dims, self.fc2_dims, self.lr, self.n_actions)
        self.reward_memory = []
        self.action_memory = []    

    def choose_action(self, state):
        state = torch.Tensor([state]).to(self.policy_function.device)
        probabilities = self.policy_function.forward(state)
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)
        return action.item()

    def store_rewards(self,reward):
        self.reward_memory.append(reward)
    
    def learn(self):
        self.policy_function.optimizer.zero_grad()

        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        G = torch.tensor(G, dtype=torch.float).to(self.policy_function.device)
        
        loss = 0
        for g, logprob in zip(G, self.action_memory):
            loss += -g * logprob
        loss.backward()
        self.policy_function.optimizer.step()

        self.action_memory = []
        self.reward_memory = []




def plot_learning_curve(scores, x, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    n_games = 3000
    agent = Agent(input_dims=[8],fc1_dims =128,fc2_dims =128 ,lr = .0005,n_actions=4,gamma =.99 )


    fname = 'REINFORCE_' + 'lunar_lunar_lr' + str(agent.lr) + '_' \
            + str(n_games) + 'games'
    figure_file =  fname + '.png'

    scores = []
    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_rewards(reward)
            observation = observation_
        agent.learn()
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score)

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(scores, x, figure_file)