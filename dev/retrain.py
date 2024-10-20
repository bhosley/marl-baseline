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

import ray
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
#from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.test_utils import (add_rllib_example_script_args,
                                        run_rllib_example_script_experiment)
from ray.tune.registry import get_trainable_cls, register_trainable
from ray.tune.stopper import (  CombinedStopper, TrialPlateauStopper,
                                MaximumIterationStopper, FunctionStopper)

from Support import get_eligible_policies, get_policy_set

# Establish depth of experimental directory (level of env in path)
#   ex. /root/test/waterworld/PPO/2_agent/ -> 3
DIR_DEPTH = 3


class MetricCallbacks(DefaultCallbacks):
    """Class for storing all of the custom metrics used in this script"""
    def on_train_result(self, *, algorithm, result: dict, **kwargs) -> None:
        result["num_agents"] = len(result['info']['learner'])
        result["episode_reward_mean"] = (
            result['env_runners']["episode_reward_mean"])


parser = add_rllib_example_script_args(
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
    "--pool-size", default=1,
    help="The best <int> or <float> proportion of policy"
    "sets to draw new policies from.",
)


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.num_agents > 0, "Must set --num-agents > 0 when training!"

    # Check validity of pool size
    try:
        args.pool_size = int(args.pool_size)
    except ValueError:
        try:
            args.pool_size = float(args.pool_size)
        except ValueError:
            print(f"{type(args.pool_size)} is an invalid pool type")

    # Ingest the provided path. A valid path will always have this first part.
    _path = args.path.split("/")
    args.prefix = ''.join(f'{_path.pop(0)}/' for _ in range(DIR_DEPTH))
    args.env = _path.pop(0)
    args.algo = _path.pop(0)
    args.trained_agents = _path.pop(0).split("_")[0]
    args.test_agents = args.num_agents

    # Check if training instance is specified
    try:
        args.training_score = float(_path.pop(0))
    except (IndexError, ValueError): # no / at end of path, or no score
        args.training_score = None

    # Get Agent Pool
    trained_pols = get_eligible_policies(args)



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


    # Use the same stoppers as baseline training.
    # Except, include a benchmark score from previous training.
    stopper = CombinedStopper(
        MaximumIterationStopper(max_iter=args.stop_iters),
        FunctionStopper(lambda trial_id, result: (
            result["num_env_steps_sampled_lifetime"] >= args.stop_timesteps)
        ),
        FunctionStopper(lambda trial_id, result: (
            result["episode_reward_mean"] >= args.stop_reward)
            # TODO: add a benchmark score
        ),
        TrialPlateauStopper(
            metric="episode_reward_mean",
            num_results=15, std=env.plateau_std
        ),
    )



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
                #module_specs={p: SingleAgentRLModuleSpec() for p in policies},
                module_specs={p: RLModuleSpec() for p in policies},
            ),
        )
        .callbacks(MetricCallbacks)
        .env_runners(num_env_runners=args.num_env_runners)
    )


    # Get a set of policies from eligible pool
    new_pols = get_policy_set(trained_pols,args.num_agents,args)
    # Build an instance of the the algorithm from the base config
    algo = base_config.build()

    # and populate with the previously trained policies
    for i in range(args.test_agents):
        algo.get_policy(f"pursuer_{i}").set_weights(new_pols[i].get_weights())
    # Register this new algorithm so that it is acessible to tune
    register_trainable("cloned_algo", lambda _: algo)


    print("starting example script")
    run_rllib_example_script_experiment(
        base_config, args, stop=stopper, trainable="cloned_algo")



