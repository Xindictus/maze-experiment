import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, chkpt_dir='tmp/sac',load_file = None):
        super(Actor, self).__init__()
       
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

        self.checkpoint_dir = chkpt_dir
        self.load_file = load_file

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        return action_logits
    
    def evaluate(self, state):
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()

        return action, dist
        
    def sample_act(self, state):
        state = torch.from_numpy(state).float().to(device)
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        argmax_action = torch.argmax(logits)
        return action.detach().cpu().numpy(), argmax_action.detach().cpu().numpy()
    


    def save_checkpoint(self,block):
        torch.save(self.state_dict(), os.path.join(self.checkpoint_dir,str(block)+'_actor.pt'))

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.load_file+'/actor.pt',map_location=device))

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, seed=1, chkpt_dir='tmp/sac',load_file = None):
        super(Critic, self).__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)


        self.checkpoint_dir = chkpt_dir
        self.load_file = load_file

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
        
    def save_checkpoint(self,block,n_critic):
        torch.save(self.state_dict(), os.path.join(self.checkpoint_dir,str(block)+'_'+n_critic+'.pt'))

    def load_checkpoint(self,n_critic):
        self.load_state_dict(torch.load(self.load_file+'/'+n_critic+'.pt',map_location=device))
    
class Value(nn.Module):
    """Value (Value) Model."""

    def __init__(self, state_size, hidden_size=32, chkpt_dir='tmp/sac',load_file = None):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        self.checkpoint_dir = chkpt_dir
        self.load_file = load_file

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def save_checkpoint(self,block):
        torch.save(self.state_dict(), os.path.join(self.checkpoint_dir,str(block)+'_value_net.pt'))

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.load_file+'/value_net.pt',map_location=device))