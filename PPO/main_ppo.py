
##only works with cartpole and lunarlander env
##olny able to solve cartpole can not solve lunarlander
##Some time throws error when back propagation saying NaN detected(exploding grad)



from memory import ReplayBuffer
from networks import ActorNetwork,CriticNetwork
import gym
import torch
from agent import Agent
import numpy as np


# env_id = 'CartPole-v0'
env_id = 'LunarLander-v2'
# env_id = 'BipedalWalker-v3'
env = gym.make(env_id).unwrapped
input_dims = env.observation_space.shape[0]
action_space = env.action_space.n

print('obs space', input_dims)
print('action space',action_space)
torch.autograd.set_detect_anomaly(True)

agent = Agent(alpha=.003, beta=.003, input_dims=input_dims, fc1_dims=128,
            fc2_dims=128,n_actions = action_space,eps_clip=.2,batch_size = 32)


game_episode_training = 1000

game_episode_testing = 10
max_step_per_episode = 3000

if __name__ == '__main__':
    for episode in range(game_episode_training):
        state = env.reset()
        total_reward = 0
        for step in range(max_step_per_episode):
            action, log_probs = agent.actor_chose_action(state)
            
            state_ , reward,done,_ = env.step(action)
            total_reward = total_reward + reward
            agent.memory.store_data_transition(state, action, reward, done, log_probs)
            state = state_


            if done:
                print('Game ------',episode,'total reward',total_reward,'total step trainned----',agent.trainning_step)
                if agent.memory.mem_cntr > agent.batch_size:
                    agent.learn()
                break


    
    
    env.seed(145)
    print('#################  finished trainning  ###############')
    for episode in range(game_episode_testing):
        state = env.reset()
        total_reward = 0
        for step in range(max_step_per_episode):
            action, _ = agent.actor_chose_action(state)
            
            state_ , reward,done,_ = env.step(action)
            env.render()
            total_reward = total_reward + reward
            # agent.memory.store_data_transition(state, action, reward, done, log_probs)
            state = state_


            if done:
                print('Game ------', episode, 'total reward', total_reward)
                break
    env.close()

  
                


















































# Total_reward_list = []


# if __name__ == '__main__':
#     for i in range(game_episode):
#         ## clearing the old experience
#         agent.memory.clear_buffer()
#         ##we have to collect trajectory
#         Collect_experience()
#         states_batch, actions_batch, rewards_batch, done_batch, old_log_batch = agent.memory.return_buffer()
#         ##compute rewards to go
#         rewards_to_go =calculate_GAE(states_batch,rewards_batch,done_batch,gamma = .95)
     

#         ##compute advantage based on value function
#         state_value = agent.critic_get_value(states_batch)
#         advantages = rewards_to_go - state_value

#         ## PPO actor critic update phase 
       
#         #find the ration of old and new policy and find min of two fnction as loss 
#         new_log_batch = agent.get_log_with_leatest_policy(states_batch,actions_batch)

#         ratio = torch.exp(new_log_batch - old_log_batch)
#         surrg1 = ratio * advantages
#         surrg2 = torch.clamp(ratio, 1 - agent.eps_clip, 1 + agent.eps_clip) * advantages

#         actor_loss = -torch.min(surrg1, surrg2).mean() 
#         agent.actor.optimizer.zero_grad()
        
#         actor_loss.backward(retain_graph=True)
#         agent.actor.optimizer.step()


#         agent.critic.optimizer.zero_grad()
#         critic_loss = torch.nn.functional.mse_loss(state_value, rewards_to_go).mean()

#         critic_loss.backward()
#         agent.critic.optimizer.step()


        



       
        
       
#         # print('haha applied gradiant')
#         # input()

        

        
#         # input()

#         if i % 10== 0:
            
#             state = env.reset()
#             done = False
#             reward_list = []
#             while done is False:
#                 action ,_= agent.actor_chose_action(state)
#                 state_, reward, done, _ = env.step(action)
                
#                 state = state_
#                 reward_list.append(reward)
#             total_episode_reward =np.sum(reward_list)
            
#             Total_reward_list.append(total_episode_reward)
#             # print('total reward list',Total_reward_list)
#             mean_total_reward =np.mean(Total_reward_list)
#             print('episode------', i ,'and reward is ', total_episode_reward )
#             # input()
        

            