import numpy as np



class ReplayBuffer():
    def __init__(self, max_memory, input_dims, n_action):
        self.mem_size = max_memory
        self.input_dims = input_dims
        self.n_action = n_action

        self.states_memory = np.zeros((self.mem_size, self.input_dims),dtype=np.float32)
        self.actions_memory = np.zeros(self.mem_size,dtype=np.int64)
        self.rewards_memory = np.zeros(self.mem_size,dtype=np.float32)
        self.done_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.log_probs = np.zeros(self.mem_size,dtype=np.float32)

        # self.new_states_memory = np.zeros((self.mem_size, self.input_dims))
        self.mem_cntr = 0

        
         

    def store_transition(self, state, action, reward, done ,log_probs):
        index = self.mem_cntr % self.mem_size
       
        self.states_memory[index] = state
        self.actions_memory[index] = action
        self.rewards_memory[index] = reward
        self.done_memory[index] = done
        self.log_probs[index] = log_probs

        # self.new_states_memory[index] = new_state
        self.mem_cntr = self.mem_cntr + 1
        
         

    def store_data_transition(self, state, action, reward, done, log_probs):
      
        index = self.mem_cntr % self.mem_size
       
        self.states_memory[index] = state
        self.actions_memory[index] = action
        self.rewards_memory[index] = reward
        self.done_memory[index] = done
        self.log_probs[index] = log_probs

        # self.new_states_memory[index] = new_state
        self.mem_cntr = self.mem_cntr + 1
        
            


    # def sample_buffer(self, batch_size):
    #     max_mem = min(self.mem_cntr, self.mem_size)
    #     batch = np.random.choice(max_mem, batch_size)
        
    #     states = self.states_memory[batch]
    #     actions = self.actions_memory[batch]
    #     rewards = self.rewards_memory[batch]
    #     dones = self.done_memory[batch]
    #     # states_ = self.new_states_memory[batch]
    #     log_probs = self.log_probs[batch]
    #     return states, actions, rewards, states_, dones



    def clear_buffer(self):
        del self.states_memory
        del self.actions_memory
        del self.rewards_memory 
        del self.done_memory 
        del self.log_probs
        # del self.new_states_memory



        self.states_memory = np.zeros((self.mem_size, self.input_dims),dtype=np.float32)
        self.actions_memory = np.zeros(self.mem_size,dtype=np.int64)
        self.rewards_memory = np.zeros(self.mem_size,dtype=np.float32)
        self.done_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.log_probs = np.zeros(self.mem_size,dtype=np.float32)

        # self.new_states_memory = np.zeros((self.mem_size, self.input_dims))
        # print('cleared buffer')
        self.mem_cntr = 0


    def return_buffer(self):
        ## return the buffer using the mem cntr

        states_batch = np.zeros((self.mem_cntr, self.input_dims), dtype=np.float32)
        actions_batch = np.zeros(self.mem_cntr,dtype=np.int64)
        rewards_batch = np.zeros(self.mem_cntr,dtype=np.float32)
        done_batch =  np.zeros(self.mem_cntr, dtype=np.float32)
        # new_states_memory = np.zeros((self.mem_cntr, self.input_dims))
        log_probs = np.zeros(self.mem_cntr,dtype=np.float32)

        for i in range(0, self.mem_cntr):
            index = i % self.mem_size

            states_batch[index] = self.states_memory[index]
            actions_batch[index] = self.actions_memory[index]
            rewards_batch[index] = self.rewards_memory[index]
            done_batch[index] = self.done_memory[index]
            log_probs[index] = self.log_probs[index]



        return states_batch,actions_batch,rewards_batch,done_batch,log_probs





 
    


        
       
