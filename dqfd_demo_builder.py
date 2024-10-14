import pandas as pd
import numpy as np
import pickle
import sys
import os
import argparse

# Load the data
def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        #print(data)
    return data

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data/data.pkl')
    parser.add_argument('--save-path', type=str, default='data/')
    return parser.parse_args()


def normalize_state(observation):
    norm_observation = [0]*len(observation)
    # x,y from -2/2 to -1/1
    norm_observation[0] = norm_feature(observation[0],-2,2)
    norm_observation[1] = norm_feature(observation[1],-2,2)
    # x,y velocity from -0/2 to 0/1
    norm_observation[2] = norm_feature(observation[2],-4,4)
    norm_observation[3] = norm_feature(observation[3],-4,4)
    # f,t from -30/30 to -1/1
    norm_observation[4] = norm_feature(observation[4],-30,30)
    norm_observation[5] = norm_feature(observation[5],-30,30)
    # # f,t velocity from -1/1 to -1/1
    norm_observation[6] = norm_feature(observation[6],-1.9,1.9)
    norm_observation[7] = norm_feature(observation[7],-1.9,1.9)

    for i in range(len(norm_observation)):
        if norm_observation[i] > 1.3:
            norm_observation[i] = 1.3
        elif norm_observation[i] < -1.3:
            norm_observation[i] = -1.3

    # print(norm_observation)
    # print("-----------------")
    # for i in range(len(observation)):
    #     if norm_observation[i] > 1.1 or norm_observation[i] < -1.1:
    #             print("ERROR")
    #             print('Feature:',i,'Value:',float(observation[i]))
    return np.array(norm_observation)

def norm_feature( feature,min,max):
    norm_feature = 2*((feature - min)/(max - min)) - 1
    return norm_feature

def get_agent_states(data,nb_blocks):
    agent_states = []
    for i in range(5):
        agent_states.append(data['block_'+str(nb_blocks)]['agent_states'])
    return agent_states
def get_dones(data,nb_blocks):
    dones = []
    for i in range(5):
        dones.append(data['block_'+str(nb_blocks)]['done'])
    return dones
class ReplayBuffer:
    """
    Convert to numpy
    """
    def __init__(self, memory_size):
        self.storage = []
        self.memory_size = memory_size
        self.next_idx = 0

    # add the samples
    def add(self, obs, action, reward, obs_, done,transition_info):
        data = (obs, action, reward, obs_, done,transition_info)
        if self.next_idx >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data
        # get the next idx
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def get_size(self):
        return len(self.storage)

    # encode samples
    def _encode_sample(self, idx,buffer_type):
        obses, actions, rewards, obses_, dones = [], [], [], [], []
        for i in idx:
            if buffer_type == 'buffer':
                data = self.storage[i]
            elif buffer_type == 'demostration':
                data = self.expert_storage[i]
            obs, action, reward, obs_, done,transition_info = data
            obses.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_.append(np.array(obs_, copy=False))
            dones.append(done)
        return np.array(obses), np.array(actions), np.array(rewards), np.array(obses_), np.array(dones),np.array(transition_info)

    # Save Buffer
    def save_buffer(self, path, name):
        path = os.path.join(path, name)
        self.storage = np.array(self.storage, dtype=object)
        np.save(path, self.storage)
    
    def get_size(self):
        return len(self.storage)


if __name__ == '__main__':
    args = arg_parse()
    data = load_data(args.path)
    games = [0,2,3,3,2,0]

    buffer = ReplayBuffer(25000)
    for i in range(1,6):
        print("Block:",i,"Game:",games[i])
        transitions = get_agent_states(data,i)
        block_dones = get_dones(data,i)
        for j in range(5):
            prev_obs = []
            real_agent_actions = []
            rewards = []
            obs = []
            dones = []
            transitions_infos = []
            print(len(transitions[0][j]))  
            for k in range(len(transitions[0][j])):
                
                    
                prev_observation = normalize_state(transitions[0][j][k]['prev_observation'])
                observation = normalize_state(transitions[0][j][k]['observation'])
                #print(prev_observation)
                action = transitions[0][j][k]['real_agent_action']
                reward = transitions[0][j][k]['reward']
                done = block_dones[0][j][k]
                transitions_info = [i,j,k]

                prev_obs.append(prev_observation)
                real_agent_actions.append(action)
                rewards.append(reward)
                obs.append(observation)
                dones.append(done)
                transitions_infos.append(transitions_info)


            # save win games to buffer
            if rewards[-1] == 10:
                if games[i] > 0:
                    for l in range(len(prev_obs)):
                        buffer.add(prev_obs[l],real_agent_actions[l],rewards[l],obs[l],dones[l],transitions_infos[l])
                    games[i] -= 1

        # If we dont have enough game to save, pass the saves to the next blocks
        if games[i] > 0:
            t = i + 1
            while games[i] > 0:
                games[t] += 1
                games[i] -= 1
                t += 1
                if t > 5:
                    t = i + 1

    print("Buffer size:",buffer.get_size())
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    buffer.save_buffer(args.save_path,'buffer.npy')
            




