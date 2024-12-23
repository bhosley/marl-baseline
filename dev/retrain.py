
# pylint: disable=fixme

from collections import deque
import numpy as np
from argparse import ArgumentParser

import ray
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.test_utils import (add_rllib_example_script_args,
                                        run_rllib_example_script_experiment)
from ray.tune.registry import get_trainable_cls, register_trainable
from ray.tune.stopper import (  CombinedStopper, TrialPlateauStopper,
                                MaximumIterationStopper, FunctionStopper)

from ray.tune import CLIReporter

from Support import get_policies_from_checkpoint

# Establish depth of experimental directory (level of env in path)
#   ex. /root/test/waterworld/PPO/2_agent/ -> 3
DIR_DEPTH = 3

parser = add_rllib_example_script_args(
    ArgumentParser(conflict_handler='resolve'), # Resolve for env
    default_iters=200,
    default_timesteps=1000000,
    default_reward=300,
)
parser.add_argument(
    "--path", type=str, default=None, required=True,
    help="absolute path to checkpoint",
)
parser.add_argument(
    "-r", "--replacement", action="store_true",
    help="Use replacement of elements in policy selection method.",
)
parser.add_argument(
    "--patience", default=10,
    help="How many iterations to continue training without improvement.",
)
parser.add_argument(
    "--env", type=str, default="waterworld",
    choices=["waterworld"],
    help="The environment to use."
    "`waterworld`: SISL WaterWorld"
    "`multiwalker`: SISL Multiwalker (Not tested yet)."
    "`pursuit`: SISL pursuit (Not tested yet).",
)


class TestCallbacks(DefaultCallbacks):
    """Class for storing all of the custom callbacks used in this script"""
    def on_train_result(self, *, algorithm, result: dict, **kwargs) -> None:
        result["num_agents"] = len(result['info']['learner'])
        result["episode_reward_mean"] = (
        result['env_runners']["episode_reward_mean"])

class CustomCallbacks(TestCallbacks):
    def on_algorithm_init(self, *, algorithm, metrics_logger = None, **kwargs) -> None:
        """Callback to do the rebuilding.""" 
        from Support import get_eligible_policies, get_policy_set

        #
        n = len(algorithm.config.policies)
        #trained_pols = get_eligible_policies(args)
        #new_pols = get_policy_set(trained_pols, n)
        new_pols = get_policies_from_checkpoint(args.path, n)

        for i in range(n):
            print("anything")
            algorithm.remove_policy(f'{env.agent_name}_{i}')
            algorithm.add_policy(f'{env.agent_name}_{i}', policy=new_pols[i])


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.num_agents > 0, "Must set --num-agents > 0 when training!"

    # Check validity of pool size
    """
    try:
        args.pool_size = int(args.pool_size)
    except ValueError:
        try:
            args.pool_size = float(args.pool_size)
        except ValueError:
            print(f"{type(args.pool_size)} is an invalid pool type")
    """

    # Ingest the provided path. A valid path will always have this first part.
    """
    _path = args.path.split("/")
    args.prefix = ''.join(f'{_path.pop(0)}/' for _ in range(DIR_DEPTH))
    args.env = _path.pop(0)
    args.algo = _path.pop(0)
    args.trained_agents = _path.pop(0).split("_")[0]
    """

    # Check if training instance is specified
    """
    try:
        args.training_score = float(_path.pop(0))
    except (IndexError, ValueError): # no / at end of path, or no score
        args.training_score = None
    """

    # Get a set of policies from eligible pool
    #trained_pols = get_eligible_policies(args)

    # Environment Switch Case
    match args.env:
        case 'waterworld':
            from Support import Waterworld
            env = Waterworld()
        case 'multiwalker':
            from Support import MultiWalker
            env = MultiWalker()
        case 'pursuit':
            from Support import Pursuit
            env = Pursuit()
    env.register(args.num_agents)

    policies = env.blank_policies(args.num_agents)

    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment( f"{args.num_agents}_agent_{args.env}" )
        .multi_agent(
            policies=policies, # Exact 1:1 mapping from AgentID to ModuleID.
            policy_mapping_fn=(lambda aid, *args, **kwargs: aid),
        )
        .rl_module(
            model_config_dict={"vf_share_layers": True},
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={p: RLModuleSpec() for p in policies},
            ),
        )
        #.evaluation(evaluation_interval=1)
        .callbacks(CustomCallbacks)
    )

    # Reimplement stopping criteria; including original max iters,
    # max timesteps, max reward, and adding plateau
    stopper = CombinedStopper(
        MaximumIterationStopper(max_iter=args.stop_iters),
        FunctionStopper(lambda trial_id, result: (
            result["num_env_steps_sampled_lifetime"] >= args.stop_timesteps)
        ),
        FunctionStopper(lambda trial_id, result: (
            result["episode_reward_mean"] >= args.stop_reward)
        ),
        TrialPlateauStopper(
            metric="episode_reward_mean",
            num_results=15, std=env.plateau_std
        ),
    )

    # Conduct the experiment
    ress = run_rllib_example_script_experiment(base_config, args, stop=stopper)

"""
python retrain3.py --path='/root/test/waterworld/PPO/4_agent/' --num-samples=2 --num-env-runners=30 --num-agents=2 --stop-iters=6
--wandb-key=913528a8e92bf601b6eb055a459bcc89130c7f5f --wandb-project=delete_me

python retrain3.py --path='/root/ray_results/PPO_2024-12-16_03-09-13/PPO_2_agent_waterworld_20755_00001_1_2024-12-16_03-09-13/checkpoint_000000' --num-samples=2 --num-env-runners=30 --num-agents=3 --stop-iters=6

python retrain3.py --path=/root/ray_results/PPO_2024-12-17_01-22-55/PPO_2_agent_waterworld_712e1_00000_0_2024-12-17_01-22-55/checkpoint_000004 --num-samples=2 --num-env-runners=30 --num-agents=3 --stop-iters=4

path=/root/ray_results/PPO_2024-12-17_01-22-55/PPO_2_agent_waterworld_712e1_00000_0_2024-12-17_01-22-55/checkpoint_000004
--num-env-runners=30 --num-agents=$a --stop-iters=4
"""