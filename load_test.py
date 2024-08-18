# %%
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig

from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.sisl import waterworld_v4

from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import register_env


"""
policies = {f"pursuer_{i}" for i in range(3)}

# Models original training:
algo_w_3_policies = (
    PPOConfig()
    .environment(
        PettingZooEnv(waterworld_v4.env())
    )
    .multi_agent(
        policies=policies,
        # Map "agent0" -> "pol0", etc...
        policy_mapping_fn=(
            #lambda agent_id, episode, worker, **kwargs: f"pursuer_{agent_id}"
            lambda aid, *args, **kwargs: aid
        ),
    )
    .build()
)
"""

def new_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "pol0" if agent_id in ["agent0", "agent1"] else "pol1"



path_to_checkpoint = "/root/ray_results/PPO_2024-07-29_14-09-46/PPO_env_35c70_00000_0_2024-07-29_14-09-46/checkpoint_000000/"
#path_to_checkpoint = "/root/ray_results/PPO_2024-07-29_14-09-46/PPO_env_35c70_00000_0_2024-07-29_14-09-46/checkpoint_000000/learner/module_state/"

# Must register the same environment 
register_env("env", lambda _: PettingZooEnv(waterworld_v4.env()))

algo_w_2_policies = Algorithm.from_checkpoint(
    checkpoint=path_to_checkpoint,
    policy_ids={"pursuer_0", "pursuer_1"},  # <- restore only those policy IDs here.
    #policy_mapping_fn=new_policy_mapping_fn,  # <- use this new mapping fn.
    policy_mapping_fn= lambda aid, *args, **kwargs: aid,  
)

# Test, whether we can train with this new setup.
out = algo_w_2_policies.train()
print(out)
# Terminate the new algo.
algo_w_2_policies.stop()





#parser = add_rllib_example_script_args(
#    default_iters=10,#200,
#    default_timesteps=1000000,
#    #default_reward=0.0,
#)
#args = parser.parse_args()


#run_rllib_example_script_experiment(algo_w_2_policies, args)