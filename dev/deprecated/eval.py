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

import wandb

from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.utils.test_utils import add_rllib_example_script_args
from ray.tune.registry import get_trainable_cls

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


if __name__ == "__main__":
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
    args.trained_agents = _path.pop(0).split("_")[0]

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

    # Pre-register each version of environment
    for n in env.agent_range:
        env.register(n)

    # Sample loop
    for _ in range(args.num_samples):
        if using_wandb:
            from ray.air.integrations.wandb import setup_wandb
            from ray.tune.utils import flatten_dict
            config={
                    'pool_size': args.pool_size,
                    'trained_agents': args.trained_agents,
                    'replacement': args.replacement,
                }
            wandb = setup_wandb(config,
                project = args.wandb_project or "Eval_Test",
                api_key=args.wandb_key)

        # Loop for agent range
        for test_agents in env.agent_range:
            policies = env.blank_policies(test_agents)

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
                    rl_module_spec=MultiRLModuleSpec(
                        rl_module_specs={p: RLModuleSpec() for p in policies},
                    ),
                )
                .evaluation(evaluation_interval=1)
            )

            algo = base_config.build()
            new_pols = get_policy_set(trained_pols,test_agents,args)
            for i in range(test_agents):
                algo.remove_policy(f'{env.agent_name}_{i}')
                algo.add_policy(f'{env.agent_name}_{i}', policy=new_pols[i])
            
            out = algo.evaluate()
            out["env_runners"]['test_agents'] = test_agents

            if using_wandb:
                flat_result = flatten_dict(out, delimiter="/")
                log = {}
                for k, v in flat_result.items():
                    log[k] = v
                wandb.log(log)

            algo.stop()
        # End of Test_agents loop

        if using_wandb:
            wandb.finish()

    # End of Sample loops
    exit()

"""
python eval.py --path='/root/test/waterworld/PPO/3_agent/' --wandb-project=delete_me --wandb-key=913528a8e92bf601b6eb055a459bcc89130c7f5f
"""