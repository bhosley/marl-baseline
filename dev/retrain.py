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

from collections import deque
import numpy as np

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

from Support import get_eligible_policies, get_policy_set

# Establish depth of experimental directory (level of env in path)
#   ex. /root/test/waterworld/PPO/2_agent/ -> 3
DIR_DEPTH = 3

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
parser.add_argument(
    "--patience", default=10,
    help="How many iterations to continue training without improvement.",
)


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.num_agents > 0, "Must set --num-agents > 0 when training!"
    args.test_agents = args.num_agents
    using_wandb = hasattr(args, "wandb_key") and args.wandb_key is not None

    # Initialize Ray.
    ray.init(
        num_cpus=args.num_cpus or None,
        local_mode=args.local_mode,
        ignore_reinit_error=True,
    )

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

    # Check if training instance is specified
    try:
        args.training_score = float(_path.pop(0))
    except (IndexError, ValueError): # no / at end of path, or no score
        args.training_score = None

    # Get a set of policies from eligible pool
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

    if using_wandb:
        import wandb
        from ray.air.integrations.wandb import setup_wandb, _is_allowed_type
        from ray.tune.utils import flatten_dict
        config={
            'algorithm': args.algo,
            'environment': args.env,
            'replacement': args.replacement,
            'pool_size': args.pool_size,
            'trained_agents': args.trained_agents,
            'test_agents': args.test_agents,
            'evaluation_freq': args.evaluation_interval,
            'checkpoint_freq': args.checkpoint_freq,
            'max_reward': args.stop_reward,
            'max_iterations': args.stop_iters,
            'max_timesteps': args.stop_timesteps,
            'patience': args.patience,
        }
        wandb = setup_wandb(config,
            project = args.wandb_project or "Retrain_Test",
            api_key = args.wandb_key)

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
    )
    if args.num_env_runners is not None:
        base_config.env_runners(num_env_runners=args.num_env_runners)
        #base_config = base_config.env_runners(
        #    num_env_runners=args.num_env_runners)

    # Build an instance of the the algorithm from the base config
    algo = base_config.build()
    # and populate with the previously trained policies
    new_pols = get_policy_set(trained_pols,args.num_agents,args)
    for i in range(args.test_agents):
        algo.remove_policy(f'{env.agent_name}_{i}')
        algo.add_policy(f'{env.agent_name}_{i}', policy=new_pols[i])

    # Tracking variables
    max_score = -np.inf
    reward_history = deque(maxlen=args.patience)
    reward_history.append(max_score)

    # Use the same stoppers as baseline training.
    def stop_check(timestep, hist, maxscore):
        """Check all stopping criteria."""
        if timestep >= args.stop_timesteps:
            return "timesteps"
        if np.mean(hist) >= args.stop_reward:
            return "reward"
        if maxscore not in hist:
            return "patience"
        return False

    for i in range(args.stop_iters):
        # Run one iteration of training
        results = algo.train()

        # Update tracking variables
        erm = results['env_runners']['episode_return_mean']
        reward_history.append(erm)
        max_score = max(max_score, erm)
        timesteps = results['env_runners']['episodes_timesteps_total']

        if using_wandb:
            flat_result = flatten_dict(results, delimiter="/")
            log = {}
            for k, v in flat_result.items():
                if _is_allowed_type(v) and not k.startswith("config/"):
                    log[k] = v
            wandb.log(log)

        # Save a checkpoint at intervals

        # Print iteration summary
        print(f"Iteration {i}: reward_mean = {erm}, "
            f"moving_average_reward = {np.mean(reward_history)}, "
            f"total_timesteps = {timesteps}, "
            f"total_episodes = {results['env_runners']['num_episodes']}")

        stop_reason = stop_check(timesteps, reward_history, max_score)
        if stop_reason:
            break

    # Out of loop criteria
    ray.shutdown()

    if using_wandb:
        wandb.log({'stop_reason': stop_reason or "Iterations"})
        wandb.finish()

    exit()

"""
tmux new-session -d \
'python retrain.py --path='/root/test/waterworld/PPO/3_agent/' --num-agents=2 --wandb-project=retrain-waterworld --wandb-key=913528a8e92bf601b6eb055a459bcc89130c7f5f'
"""
