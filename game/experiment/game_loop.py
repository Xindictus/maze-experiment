import time

from game.game_utils import get_distance_traveled


def run_game_block(
    env,
    agent_callback,
    step_callback,
    block_name,
    block_number,
    max_games,
    max_game_duration,
    action_duration,
    popup_window_time,
    games_per_block,
    mode,
    log_action,
    can_learn=True,
):
    block_metrics = {
        "fps": [],
        "actions": [],
        "distance_travel": [],
        "agent_states": [],
        "step_duration": [],
        "game_duration": [],
        "step_counter": [],
        "rewards": [],
        "states": [],
        "game_score": [],
        "win_loss": [],
        "done": [],
    }

    train_game_success_counter = 0

    for i_game in range(max_games):
        is_on_pause = True
        while is_on_pause:
            prev_obs, _, is_on_pause = env.reset(mode)
            norm_prev_obs = agent_callback.normalize_state(prev_obs)

        game_reward = 200
        distance_travel = 0
        redundant_end = 0
        game_start = time.time()
        game_done_flags = []
        step_durations = []
        game_fps = []
        game_actions = []
        game_rewards = []
        game_states = []
        agent_states = []
        step_counter = 0

        while True:
            step_counter += 1
            if time.time() - game_start - redundant_end >= max_game_duration:
                timed_out = True
            else:
                timed_out = False

            env_action, real_action = agent_callback.get_action(
                norm_prev_obs, mode
            )
            transition = env.step(
                env_action,
                timed_out,
                action_duration,
                mode=mode,
                text=log_action(i_game, max_games),
            )

            (obs, reward, done, fps, pause_dur, action_pair, _) = transition
            norm_obs = agent_callback.normalize_state(obs)
            redundant_end += pause_dur

            step_durations.append(time.time() - game_start - pause_dur)
            game_fps.append(fps)
            game_actions.append(action_pair)
            game_rewards.append(reward)
            game_states.append(obs)
            game_done_flags.append(done)
            distance_travel = get_distance_traveled(
                distance_travel, prev_obs, obs
            )

            agent_states.append(
                step_callback.create_log_row(
                    prev_obs, obs, env_action, action_pair, reward, real_action
                )
            )

            if mode == "train":
                step_callback.save_experience(
                    norm_prev_obs,
                    norm_obs,
                    real_action,
                    reward,
                    done,
                    block_number,
                    i_game,
                    step_counter,
                )

            prev_obs = obs
            norm_prev_obs = norm_obs
            game_reward += reward

            if done:
                redundant_end += popup_window_time
                if not timed_out:
                    train_game_success_counter += 1
                else:
                    game_reward = 0
                time.sleep(popup_window_time)
                break

        # Accumulate metrics
        block_metrics["done"].append(game_done_flags)
        block_metrics["step_duration"].append(step_durations)
        block_metrics["fps"].append(game_fps)
        block_metrics["actions"].append(game_actions)
        block_metrics["rewards"].append(game_rewards)
        block_metrics["states"].append(game_states)
        block_metrics["distance_travel"].append(distance_travel)
        block_metrics["agent_states"].append(agent_states)
        block_metrics["step_counter"].append(step_counter)
        block_metrics["game_score"].append(game_reward)
        block_metrics["game_duration"].append(
            time.time() - game_start - redundant_end
        )

        # Callbacks for post-game updates
        agent_callback.after_game(game_reward)

        if can_learn and agent_callback.should_learn(
            mode, block_number, i_game
        ):
            agent_callback.learn(
                i_game, block_number, block_metrics[block_name]
            )

    block_metrics["win_loss"] = [
        train_game_success_counter,
        games_per_block - train_game_success_counter,
    ]

    return block_metrics
