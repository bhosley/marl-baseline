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
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
#from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
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
        wandb.init(project = args.wandb_project or "Retrain_Test",
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
        )

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
                #module_specs={p: SingleAgentRLModuleSpec() for p in policies},
                module_specs={p: RLModuleSpec() for p in policies},
            ),
        )
        .evaluation(evaluation_interval=1)
    )
    #if args.num_env_runners:
    #    base_config = base_config.env_runners(
    #        num_env_runners=args.num_env_runners)



    # Build an instance of the the algorithm from the base config
    algo = base_config.build()


    new_pols = get_policy_set(trained_pols,args.num_agents,args)
    # and populate with the previously trained policies
    for i in range(args.test_agents):
        algo.get_policy(f"pursuer_{i}").set_weights(new_pols[i].get_weights())





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

        # Evaluate at intervals
        eva = algo.evaluate()
        print(f"Evaluation {i}: reward_mean = {eva['env_runners']['episode_return_mean']}")


        # Run one iteration of training
        results = algo.train()

        # Update tracking variables
        erm = results['env_runners']['episode_return_mean']
        reward_history.append(erm)
        max_score = max(max_score, erm)
        timesteps = results['env_runners']['episodes_timesteps_total']

        if using_wandb:
            wandb.log(results["env_runners"])



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
    if using_wandb:
        wandb.log({'stop_reason': stop_reason or "Iterations"})
        wandb.finish()



    exit()

"""
python retrain.py --stop-iters=10 --path='/root/test/waterworld/PPO/2_agent/' --num-agents=4 \
--wandb-project=delete_me_2 --wandb-key=913528a8e92bf601b6eb055a459bcc89130c7f5f


        # Save a checkpoint at intervals
        if i % args.checkpoint_freq == 0:
            checkpoint = algo.save()
            print(f"Checkpoint saved at iteration {i}: {checkpoint}")

        # Evaluate at intervals
        if i % args.evaluation_freq == 0:
            eval_results = algo.evaluate()
            if using_wandb:
                wandb.log(eval_results)
            print(f"Evaluation results at iteration {i}: {eval_results}")


# Initialize Weights and Biases (optional)

checkpoint_freq = 10   # How often to save checkpoints
evaluation_freq = 5    # How often to evaluate

# Stopping criteria
max_episodes = 1000    
max_timesteps = 100000  
reward_threshold = 200  
moving_average_window = 10  
min_reward_improvement = 1e-3  


for i in range(num_iterations):
    # Run one iteration of training
    results = algo.train()
    
    # Update tracking variables
    total_episodes += results['episodes_this_iter']
    total_timesteps += results['timesteps_this_iter']
    






    # Register this new algorithm so that it is acessible to tune
    register_trainable("cloned_algo", lambda _: algo)

    print("starting example script")
    run_rllib_example_script_experiment(
        base_config, args, stop=stopper, trainable="cloned_algo")


"""
