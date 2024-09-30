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

from glob import glob
import numpy as np
import wandb

from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.test_utils import add_rllib_example_script_args
from ray.tune.registry import get_trainable_cls, register_env

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



def get_eligible_policies(args):
    """Return a list of policy objects based on parsed args"""
    recon_path = f'{args.prefix}{args.env}/{args.algo}/{args.num_agents}_agent/'
    if args.training_score:
        recon_path += str(args.training_score)

    # Sort by descending value
    roster = sorted(glob(recon_path+"*"), reverse=True)

    # Number of training instances to pool
    if isinstance(args.pool_size, int):
        num = args.pool_size
    else: # then it must be a float
        num = -int( args.pool_size * len(roster) // -1 ) # Rounded Up

    # Get policies from checkpoints from each training instance in roster
    pols = [Policy.from_checkpoint(p)
            for i in roster[:num]
            for p in glob(f"{i}/policies/*")]
    return pols


def get_policy_set(pols,n, args):
    """Takes pretrained policy set, and returns a new set of n-length"""
    if args.replacement:
        return np.random.choice(pols,n)
    else:
        new_set = []
        while len(new_set) < n:
            dif = min(len(pols),n-len(new_set))
            for e in np.random.choice(pols, dif, replace=False):
                new_set.append(e)
        return new_set



def main():
    """Main defined as a function global scoping of variables"""
    args = parser.parse_args()
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
        # pylint: disable=C0415

        case 'multiwalker':
            raise NotImplementedError("This environment not yet implemented")

        case 'pursuit':
            raise NotImplementedError("This environment not yet implemented")

        case 'waterworld':
            from pettingzoo.sisl import waterworld_v4
            agent_range = range(2,9)
            # Pre-register each version of environment
            # TODO: Is this the best way?
            for a in agent_range:
                register_env(f"{a}_agent_waterworld", lambda _:
                    ParallelPettingZooEnv(
                        waterworld_v4.parallel_env(n_pursuers=a)))
            def blank_policies(n):
                return {f"pursuer_{i}" for i in range(n)}


    # Sample loop
    for _ in range(args.num_samples):
        if using_wandb:
            wandb.init(project = args.wandb_project or "Eval_Test")

        # Loop for agent range
        for test_agents in agent_range:
            policies = blank_policies(n=test_agents)

            base_config = (
                get_trainable_cls(args.algo)
                .get_default_config()
                .environment( f"{test_agents}_agent_{args.env}" )
                .multi_agent(
                    policies=policies,
                    policy_mapping_fn=(lambda aid, *args, **kwargs: aid),
                )
                .rl_module(
                    model_config_dict={"vf_share_layers": True},
                    rl_module_spec=MultiAgentRLModuleSpec(
                        module_specs={
                            p: SingleAgentRLModuleSpec() for p in policies},
                    ),
                )
                .evaluation(evaluation_interval=1)
            )

            algo = base_config.build()
            new_pols = get_policy_set(trained_pols,test_agents,args)
            for i in range(test_agents):
                algo.get_policy(f"pursuer_{i}").set_weights(new_pols[i].get_weights())

            out = algo.evaluate()
            out["env_runners"]['test_agents'] = test_agents
            out["env_runners"]['trained_agents'] = args.num_agents
            out["env_runners"]['replacement'] = args.replacement
            out["env_runners"]['pool_size'] = args.pool_size

            if using_wandb:
                wandb.log(out["env_runners"])

            algo.stop()
        # End of Test_agents loop

        if using_wandb:
            wandb.finish()

    # End of Sample loops


if __name__ == "__main__":
    main()
    exit()
