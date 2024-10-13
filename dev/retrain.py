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

from ray.air.constants import TRAINING_ITERATION
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.utils.metrics import (
        ENV_RUNNER_RESULTS,
        EPISODE_RETURN_MEAN,
        NUM_ENV_STEPS_SAMPLED_LIFETIME,
    )
from ray.rllib.utils.test_utils import add_rllib_example_script_args
from ray.tune import CLIReporter
from ray.tune.registry import get_trainable_cls
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
    args.num_agents = _path.pop(0).split("_")[0]

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
        .env_runners(num_env_runners=args.num_env_runners)
    )


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

    # Log results using WandB.
    tune_callbacks = []
    if hasattr(args, "wandb_key") and (args.wandb_key is not None):
        wandb_key = args.wandb_key
        project = args.wandb_project or (
            args.algo.lower() + "-" + str(env.env_name).lower()
        )
        tune_callbacks.append(
            WandbLoggerCallback(
                api_key=wandb_key,
                project=project,
                upload_checkpoints=True,
                **({"name":args.wandb_run_name} if args.wandb_run_name else {}),
            )
        )


    # Auto-configure a CLIReporter (to log the results to the console).
    # Use better ProgressReporter for multi-agent cases: List individual policy rewards.
    progress_reporter = CLIReporter(
        metric_columns={
            **{
                TRAINING_ITERATION: "iter",
                "time_total_s": "total time (s)",
                NUM_ENV_STEPS_SAMPLED_LIFETIME: "ts",
                f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": "combined return",
            },
            **{
                (
                    f"{ENV_RUNNER_RESULTS}/module_episode_returns_mean/" f"{pid}"
                ): f"return {pid}"
                for pid in base_config.policies
            },
        },
    )








    # Run the actual experiment (using Tune).
    start_time = time.time()
    results = tune.Tuner(
        trainable or config.algo_class,
        param_space=config,
        run_config=air.RunConfig(
            stop=stop,
            verbose=args.verbose,
            callbacks=tune_callbacks,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=args.checkpoint_freq,
                checkpoint_at_end=args.checkpoint_at_end,
            ),
            progress_reporter=progress_reporter,
        ),
        tune_config=tune.TuneConfig(
            num_samples=args.num_samples,
            max_concurrent_trials=args.max_concurrent_trials,
            scheduler=scheduler,
        ),
    ).fit()
    time_taken = time.time() - start_time

    ray.shutdown()

