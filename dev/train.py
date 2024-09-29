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

"""

# pylint: disable=fixme

from argparse import ArgumentParser

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.utils.test_utils import (add_rllib_example_script_args,
                                        run_rllib_example_script_experiment)
from ray.tune.registry import get_trainable_cls, register_env
from ray.tune.stopper import (  CombinedStopper, TrialPlateauStopper,
                                MaximumIterationStopper, FunctionStopper)


class MetricCallbacks(DefaultCallbacks):
    """Class for storing all of the custom metrics used in this script"""
    def on_train_result(self, *, algorithm, result: dict, **kwargs) -> None:
        result["num_agents"] = len(result['env_runners']['agent_steps'])
        result["episode_reward_mean"] = (
            result['env_runners']["episode_reward_mean"])


parser = add_rllib_example_script_args(
    ArgumentParser(conflict_handler='resolve'), # Resolve for env
    default_iters=200,
    default_timesteps=1000000,
    default_reward=300,
)
parser.add_argument(
    "--env", type=str, default="waterworld",
    choices=["waterworld"],
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
            from pettingzoo.sisl import waterworld_v4
            register_env(f"{args.num_agents}_agent_{args.env}", lambda _:
                ParallelPettingZooEnv(
                    waterworld_v4.parallel_env(n_pursuers=args.num_agents)))
            policies = {f"pursuer_{i}" for i in range(args.num_agents)}

        case 'multiwalker':
            raise NotImplementedError("This environment not yet implemented")
            #register_env(f"{args.num_agents}_agent_{args.env}", lambda _:
            #   PettingZooEnv(multiwalker_v9.env()))
            #policies = {f"walker_{i}" for i in range(args.num_agents)}

        case 'pursuit':
            raise NotImplementedError("This environment not yet implemented")
            #register_env(f"{args.num_agents}_agent_{args.env}", lambda _:
            #   PettingZooEnv(pursuit_v4.env()))
            #policies = {f"pursuer_{i}" for i in range(args.num_agents)}

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
            rl_module_spec=MultiAgentRLModuleSpec(
                module_specs={p: SingleAgentRLModuleSpec() for p in policies},
            ),
        )
        .callbacks(MetricCallbacks)
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
            num_results=15, std=3
        ),
    )

    # Conduct the experiement
    ress = run_rllib_example_script_experiment(base_config, args, stop=stopper)

    # Organize results
    if args.checkpoint_at_end:
        import shutil
        for i, res in enumerate(ress):
            source = res.path + "/checkpoint_000000"
            score = res.metrics['env_runners']['episode_reward_mean']
            dest = (f"/root/test/{args.env}/{args.algo}/" +
                    f"{args.num_agents}_agent/{score}")
            shutil.move(source, dest, copy_function = shutil.copytree)

    exit()