import sys
import time

from tqdm import tqdm


def offline_grad_updates_session(
    agent, second_agent, config, mode, block_number
):
    """
    Perform a number of offline gradient updates and optionally save models.
    """
    start_time = time.time()
    print("Starting offline gradient updates session")

    update_cycles = config["Experiment"][mode]["updates_per_ogu"]
    if update_cycles > 0:
        grad_updates(
            agent, second_agent, config, mode, update_cycles, block_number + 1
        )

        print("Saving model")
        agent.save_models(block_number)
        if mode == "no_tl_two_agents":
            print("Saving second model")
            second_agent.save_models(block_number)

    return time.time() - start_time


def grad_updates(
    agent, second_agent, config, mode, update_cycles, block_number
):
    """
    Performs a number of offline gradient updates on the agent(s).
    """
    start_time = time.time()
    print(f"Performing {update_cycles} gradient updates...")

    for cycle_i in tqdm(range(update_cycles), file=sys.stdout):
        if not config["SAC"].get("freeze_agent", False):
            loss = agent.learn(block_number)
            if cycle_i % 100 == 0:
                _print_losses("Agent", cycle_i, loss)

        if mode == "no_tl_two_agents" and not config["SAC"].get(
            "freeze_second_agent", False
        ):
            loss = second_agent.learn(block_number)
            if cycle_i % 100 == 0:
                _print_losses("Second Agent", cycle_i, loss)

    return time.time() - start_time


def pre_train_agent(agent, total_updates, second_human=False):
    if second_human:
        return

    print(f"Starting supervised pre-training for {total_updates} steps.")
    for cycle_i in tqdm(range(total_updates), file=sys.stdout):
        loss = agent.supervised_learn(0)
        if cycle_i % 1000 == 0:
            print(f"Cycle: {cycle_i} | Loss: {loss}")

    agent.soft_update_target()


def _print_losses(label, cycle, loss_tuple):
    policy_loss, q1_loss, q2_loss, alpha = loss_tuple
    print(
        f"[{label}] Cycle {cycle}"
        f" | Policy: {policy_loss:.4f}"
        f" | Q1: {q1_loss:.4f}"
        f" | Q2: {q2_loss:.4f}"
        f" | Alpha: {alpha:.4f}"
    )
