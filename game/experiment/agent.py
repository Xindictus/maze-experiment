import random

import numpy as np

from game.game_utils import get_agent_only_action, get_env_action


def get_agent_action(
    prev_observation,
    *,
    agent,
    second_agent=None,
    second_human=False,
    is_discrete=True,
    enemy_status=None,
    game_mode="train",
    block=None,
    use_ppr=False,
    ppr_table=None
):
    """
    Determines the agent action and returns both the environment-compatible
    and internal representations.
    """
    if second_human:
        return None, None

    if enemy_status == "agent":
        agent_action, argmax_action = compute_agent_action(
            prev_observation, agent, second_agent, block, use_ppr, ppr_table
        )
        return _finalize_action(
            agent_action, argmax_action, is_discrete, game_mode
        )

    elif enemy_status == "second_agent":
        agent_action, argmax_action = compute_agent_action(
            prev_observation, second_agent, None, block, False, None
        )
        return _finalize_action(
            agent_action, argmax_action, is_discrete, game_mode
        )

    elif enemy_status == "only_agent":
        agent_action, argmax_action = compute_agent_action(
            prev_observation, agent, second_agent, block, use_ppr, ppr_table
        )
        multi_actions = get_agent_only_action(
            agent_action if game_mode == "train" else argmax_action
        )
        return (
            [get_env_action(a, is_discrete) for a in multi_actions],
            agent_action,
        )

    elif enemy_status == "random":
        action = int(np.random.choice([0, 1, 2]))
        return get_env_action(action, is_discrete), action

    return None, None


def _finalize_action(agent_action, argmax_action, is_discrete, game_mode):
    if game_mode == "train":
        return get_env_action(agent_action, is_discrete), agent_action
    else:
        return get_env_action(argmax_action, is_discrete), argmax_action


def compute_agent_action(
    observation,
    agent,
    second_agent=None,
    block=None,
    use_ppr=False,
    ppr_table=None,
):
    """
    Computes action from an agent, with optional policy reuse.
    """
    action = agent.actor.sample_act(observation)
    argmax = action

    if use_ppr and second_agent and block is not None and ppr_table:
        expert_action = second_agent.actor.sample_act(observation)
        if random.random() < ppr_table[block]:
            action = expert_action
            argmax = expert_action

    return action, argmax


def save_experience(
    interaction, agent, second_agent, config, mode, second_human
):
    if config["Experiment"]["agent"] != "sac" or second_human:
        return

    obs, actions, reward, obs_next, done, transition_info = interaction
    agent.memory.add(obs, actions[0], reward, obs_next, done, transition_info)

    if mode == "no_tl_two_agents":
        second_agent.memory.add(
            obs, actions[1], reward, obs_next, done, transition_info
        )
