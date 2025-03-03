"""Runs env in RLlib 

How to run this script
----------------------
`python [script file name].py --num-agents=2`

For debugging, use the following additional command line options
`--no-tune --num-env-runners=0`
which should allow you to set breakpoints anywhere in the RLlib code and
have the execution stop there for inspection and debugging.

For logging to your WandB account, use:
`--wandb-key=[your WandB API key] --wandb-project=[some project name]
--wandb-run-name=[optional: WandB run name (within the defined project)]`

tmux new-session -d 'python train.py --checkpoint-at-end --num-samples=10 --num-env-runners=30\
--wandb-key=913528a8e92bf601b6eb055a459bcc89130c7f5f --wandb-project=mini-test-waterworld --num-agents=4'
"""

# pylint: disable=fixme

from argparse import ArgumentParser

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.test_utils import (add_rllib_example_script_args,
                                        run_rllib_example_script_experiment)
from ray.tune.registry import get_trainable_cls
from ray.tune.stopper import (  CombinedStopper, TrialPlateauStopper,
                                MaximumIterationStopper, FunctionStopper)


class MetricCallbacks(DefaultCallbacks):
    """Class for storing all of the custom metrics used in this script"""
    def on_train_result(self, *, algorithm, result: dict, **kwargs) -> None:
        try:
            result["num_agents"] = len(result['info']['learner'])
            result["episode_reward_mean"] = (
                result['env_runners']["episode_reward_mean"])
        except:
            pass

parser = add_rllib_example_script_args(
    ArgumentParser(conflict_handler='resolve'), # Resolve for env
    default_iters=200,
    default_timesteps=1000000,
    default_reward=300,
)
parser.add_argument(
    "--env", type=str, default="waterworld",
    choices=["waterworld","multiwalker","pursuit"],
    help="The environment to use."
    "`waterworld`: SISL WaterWorld"
    "`multiwalker`: SISL Multiwalker (Not tested yet)."
    "`pursuit`: SISL pursuit (Not tested yet).",
)

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
        .environment( f"{args.num_agents}_agent_{args.env}",
            disable_env_checking = True    
        )
        .multi_agent(
            policies=policies, # Exact 1:1 mapping from AgentID to ModuleID.
            policy_mapping_fn=(lambda aid, *args, **kwargs: aid),
        )
        .rl_module(
            model_config_dict={
                "vf_share_layers": True,
                #"fcnet_hiddens": [256, 256, 256, 256, 256, 256],
                #6
                #"fcnet_hiddens": [400, 300],
                #"fcnet_activation": "relu",
                #"use_attention": True,
                #8
                #"fcnet_hiddens": [256, 256, 256, 256, 256, 256],
                #
                # 1
                #
                #"vf_share_layers": True,
                #"fcnet_hiddens": [400, 300],
                #"fcnet_activation": "relu",
                #"use_attention": True,
                #
                #"use_lstm": True,
                # 
                # 2 (trying Lower KL)
                #"fcnet_hiddens": [256, 256, 256, 256, 256, 256],
                #"kl_coeff": 0.1,
                # 3 Increase entropy to increase exploration
                #"entropy_coeff": 0.01,
                #"fcnet_hiddens": [256, 256, 256],
                # 4
                # 5 Shorten/widen nets, add relu, add attention
                #"fcnet_hiddens": [400, 300],
                #"fcnet_activation": "relu",
                #"use_attention": True, 
                # 
                #"attention_head_dim": 32
                # 6 LSTM?
                #"use_lstm": True,
                # 
                # 7 Trying SAC Again
                "fcnet_hiddens": [400, 300],
                },
            rl_module_spec=MultiRLModuleSpec( 
                rl_module_specs={p: RLModuleSpec() for p in policies},
            ),
        )
        .callbacks(MetricCallbacks)
        #.training
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

    # Record information
    base_config["num_agents"] = args.num_agents
    base_config["steps_pretrained"] = 0

    # Conduct the experiment
    ress = run_rllib_example_script_experiment(base_config, args, stop=stopper)

    # Organize results
    if args.checkpoint_at_end:
        import shutil
        for i, res in enumerate(ress):
            source = res.path + "/checkpoint_000000"
            score = res.metrics['env_runners']['episode_reward_mean']
            dest = (f"/root/test/{args.env}/{args.algo}/" +
                    f"{args.num_agents}_agent/{score}")
            shutil.copytree(source, dest, dirs_exist_ok=True)

    for res in ress: 
        print(res.path)
    exit()

"""
python train.py --num-samples=2 --num-env-runners=30 --num-agents=2 --stop-iters=10 --checkpoint-freq=2

python train.py --num-samples=2 --env='multiwalker' --num-env-runners=30 --num-agents=3 --stop-iters=10 --checkpoint-freq=2

tmux new-session -d "python train.py --env='multiwalker' --checkpoint-at-end --num-samples=10 --num-env-runners=30 --wandb-key=913528a8e92bf601b6eb055a459bcc89130c7f5f --wandb-project=multiwalker --num-agents=4"

tmux new-session -d "python train.py --env='multiwalker' --checkpoint-at-end --num-samples=1 --num-env-runners=30 --wandb-key=913528a8e92bf601b6eb055a459bcc89130c7f5f --wandb-project=multiwalker --num-agents=4 --stop-iters=500 --stop-timesteps=100000000"

tmux new-session -d "python train.py --env='multiwalker' --num-samples=1 --num-env-runners=30 --wandb-key=913528a8e92bf601b6eb055a459bcc89130c7f5f --wandb-project=tune_multiwalker --num-agents=4 --stop-iters=500 --stop-timesteps=100000000"

"""