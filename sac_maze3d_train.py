# Virtual environment
import numpy as np
from maze3D_new.Maze3DEnvRemote import Maze3D as Maze3D_v2
# from maze3D_new.assets import *
# from maze3D_new.utils import save_logs_and_plot

# Experiment
from game.experiment import Experiment
from game.game_utils import  get_config
# RL modules
from rl_models.utils import get_sac_agent

import sys
import time
from datetime import timedelta
import argparse
import torch
import os
from prettytable import PrettyTable
from datetime import timedelta
from game.arguments import get_game_args

"""
The code of this work is based on the following github repos:
https://github.com/kengz/SLM-Lab
https://github.com/EveLIn3/Discrete_SAC_LunarLander/blob/master/sac_discrete.py
"""

def print_setting(agent,x):
    ID,actor_lr,critic_lr,alpha_lr,hidden_size,tau,gamma,batch_size,target_entropy,log_alpha,freeze_status = agent.return_settings()
    x.field_names = ["Agent ID", "Actor LR", "Critic LR", "Alpha LR", "Hidden Size", "Tau", "Gamma", "Batch Size", "Target Entropy", "Log Alpha", "Freeze Status"]
    x.add_row([ID,actor_lr,critic_lr,alpha_lr,hidden_size,tau,gamma,batch_size,target_entropy,log_alpha,freeze_status])
    return x

def check_save_dir(config,participant_name):
    checkpoint_dir = config['SAC']['chkpt']
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    if os.path.isdir(os.path.join(checkpoint_dir,participant_name)) == False:
        os.mkdir(os.path.join(checkpoint_dir,participant_name))
        checkpoint_dir = os.path.join(checkpoint_dir,participant_name) + '/'
    else:
        c = 1
        while os.path.isdir(os.path.join(checkpoint_dir,participant_name+str(c))) == True:
            c += 1
        os.mkdir(os.path.join(checkpoint_dir,participant_name+str(c)))
        checkpoint_dir = os.path.join(checkpoint_dir,participant_name+str(c)) + '/'

    config['SAC']['chkpt'] = checkpoint_dir
    return config

def main(argv):
    args = get_game_args()
    # get configuration
    print('IM trying to get this config')
    print(args.config)
    
    config = get_config(args.config)
    
    print('Config loaded',config)

    # creating environment
    maze = Maze3D_v2(config_file=args.config)
    loop = config['Experiment']['mode']
    if loop != 'human':
        config = check_save_dir(config,args.participant)
        print_array = PrettyTable()
        if args.agent_type == "basesac":
            agent = get_sac_agent(args,config, maze,p_name=args.participant,ID='First')
            agent.save_models('Initial')
        print_array = print_setting(agent,print_array)

        if loop == 'no_tl_two_agents':
            if args.agent_type == "basesac":
                second_agent = get_sac_agent(args,config, maze, p_name=args.participant,ID='Second')
                second_agent.save_models('Initial')
            print_array = print_setting(second_agent,print_array)
        else:
            second_agent = None

        print('Agent created')
        print(print_array)
        # create the experiment
    else:
        agent = None
        second_agent = None
    experiment = Experiment(maze, agent, config=config,participant_name=args.participant,second_agent=second_agent)

    start_experiment = time.time()

    # Run a Pre-Training with Expert Buffers
    
    if args.Load_Expert_Buffers or args.dqfd:
        print('Loading Buffer \n DQfD:',args.dqfd,'\n Expert Buffers:',args.Load_Expert_Buffers)
        experiment.test_buffer(2500)
        


    if loop == 'no_tl':
        experiment.mz_experiment(args.participant)
    elif loop == 'no_tl_only_agent':
        experiment.mz_only_agent(args.participant)
    elif loop == 'no_tl_two_agents':
        experiment.mz_two_agents(args.participant)
    elif loop == 'eval':
        experiment.mz_eval(args.participant)
    elif loop == 'human':
        experiment.human_play(args.participant)
    else:
        print("Unknown training mode")
        exit(1)
    #experiment.env.finished()
    end_experiment = time.time()
    experiment_duration = timedelta(seconds=end_experiment - start_experiment - experiment.duration_pause_total)
    
    print('Total Experiment time: {}'.format(experiment_duration))

    return


if __name__ == '__main__':
    main(sys.argv[1:])
    exit(0)