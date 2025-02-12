from argparse import ArgumentParser
from glob import glob

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.test_utils import (add_rllib_example_script_args,
                                        run_rllib_example_script_experiment)
from ray.tune.registry import get_trainable_cls
from ray.tune.stopper import (  CombinedStopper, TrialPlateauStopper,
                                MaximumIterationStopper, FunctionStopper)


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
    choices=["waterworld","multiwalker","pursuit"],
    help="The environment to use."
    "`waterworld`: SISL WaterWorld"
    "`multiwalker`: SISL Multiwalker (Not tested yet)."
    "`pursuit`: SISL pursuit (Not tested yet).",
)
parser.add_argument(
    "--steps_pretrained", type=int, default=0,
    help="The number of iterations pretrained before this script."
)


class CustomCallbacks(DefaultCallbacks):
    """Class for storing all of the custom callbacks used in this script"""
    def on_algorithm_init(self, *, algorithm, **kwargs) -> None:
        """Callback to do the rebuilding.""" 
        from Support import get_policies_from_checkpoint

        n = len(algorithm.config.policies)
        #new_pols = get_policies_from_checkpoint(args.path, n)
        new_pols = env.get_policies_from_checkpoint(args.path, n)
        for i in range(n):
            algorithm.remove_policy(f'{env.agent_name}_{i}')
            algorithm.add_policy(f'{env.agent_name}_{i}', policy=new_pols[i])


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.num_agents > 0, "Must set --num-agents > 0 when training!"

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
        .callbacks(CustomCallbacks)
    )

    if args.env is 'multiwalker':
        base_config = base_config.training(
            model={
                "fcnet_hiddens": [256, 256, 256, 256, 256, 256],
                "fcnet_activation": "relu",
            },
        )

    # Record information
    base_config["steps_pretrained"] = args.steps_pretrained
    base_config["num_agents"] = args.num_agents
    base_config["num_pretrained_agents"] = len(glob(f"{args.path}/policies/*"))

    # Reimplement stopping criteria; and add attention as plateau 
    stopper = CombinedStopper(
        MaximumIterationStopper(max_iter=args.stop_iters),
        FunctionStopper(lambda trial_id, result: (
            result["num_env_steps_sampled_lifetime"] >= args.stop_timesteps)
        ),
        FunctionStopper(lambda trial_id, result: (
            result['env_runners']["episode_reward_mean"] >= args.stop_reward)    
        ),
        TrialPlateauStopper(
            metric="env_runners/episode_reward_mean",
            num_results=15, std=env.plateau_std
        ),
    )

    # Conduct the experiment
    ress = run_rllib_example_script_experiment(base_config, args, stop=stopper)

"""
python retrain.py \
--path=/root/ray_results/PPO_2024-12-17_01-22-55/PPO_2_agent_waterworld_712e1_00000_0_2024-12-17_01-22-55/checkpoint_000004 \
--num-env-runners=30 --num-agents=2 --stop-iters=4 --steps_pretrained=10
--wandb-key=913528a8e92bf601b6eb055a459bcc89130c7f5f --wandb-project=delete_me_2 \
"""