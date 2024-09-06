from random import choice

from pettingzoo.sisl import waterworld_v4
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import register_env


parser = add_rllib_example_script_args(
    default_iters=1, default_reward=300, default_timesteps=50000)

parser.add_argument(
    "--reps", type=int, default=1,
    help="The number of repetitions of the experiment",
)


#path_to_checkpoint = "/root/ray_results/PPO_2024-07-29_14-09-46/PPO_env_35c70_00000_0_2024-07-29_14-09-46/checkpoint_000000/" 
#trained_agents = 3

path_to_checkpoint = "/root/ray_results/PPO_2024-07-29_14-38-07/PPO_env_2b8f1_00000_0_2024-07-29_14-38-07/checkpoint_000000"
trained_agents = 4

#path_to_checkpoint = "/root/ray_results/PPO_2024-07-29_15-06-00/PPO_env_10dfa_00000_0_2024-07-29_15-06-00/checkpoint_000000"
#trained_agents = 5

#path_to_checkpoint = "/root/ray_results/PPO_2024-07-29_15-37-27/PPO_env_75c7d_00000_0_2024-07-29_15-37-27/checkpoint_000000"
#trained_agents = 6

#path_to_checkpoint = "/root/ray_results/PPO_2024-07-29_16-06-40/PPO_env_8ab1e_00000_0_2024-07-29_16-06-40/checkpoint_000000"
#trained_agents = 7

#path_to_checkpoint = "/root/ray_results/PPO_2024-07-29_16-35-04/PPO_env_81eea_00000_0_2024-07-29_16-35-04/checkpoint_000000"
#trained_agents = 8


# Must register the same environment 
register_env("env", lambda _: PettingZooEnv(waterworld_v4.env()))

def new_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return agent_id if agent_id in trained_pols else choice(trained_pols)

import wandb
import json

if __name__ == "__main__":
    args = parser.parse_args() 
    wandb.init(project="Eval_Test")

    for _ in range(args.reps):
        trained_pols = {f"pursuer_{i}" for i in range(trained_agents)}

        for test_agents in range(8):
            policies = {f"pursuer_{i}" for i in range(test_agents)}

            restored_algo = Algorithm.from_checkpoint(
                checkpoint=path_to_checkpoint,
                policy_ids=policies,  # <- restore only those policy IDs here.
                policy_mapping_fn=new_policy_mapping_fn,  # <- use this new mapping fn.
            )

            # Test, whether we can train with this new setup.
            out = restored_algo.train()
            out["env_runners"]['test_agents'] = test_agents
            out["env_runners"]['trained_agents'] = trained_agents

            # Log test results
            wandb.log(out["env_runners"])

            # Terminate the new algo.
            restored_algo.stop()



    






