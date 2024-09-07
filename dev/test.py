"""Runs env in RLlib 

Current running config
----------------------
```
python test.py --wandb-project=mini_test_ww2 \
--wandb-key=913528a8e92bf601b6eb055a459bcc89130c7f5f \
--num-samples=10 --num-env-runners=30 --checkpoint-at-end --num-agents=3
```

python test.py --wandb-project=eval_test_2 \
--wandb-key=913528a8e92bf601b6eb055a459bcc89130c7f5f \
--num-env-runners=30 --checkpoint-at-end --num-agents=3

python test.py --output /root/scrap_outputs --num-agents=2


How to run this script
----------------------
`python [script file name].py --enable-new-api-stack --num-agents=2`

For debugging, use the following additional command line options
`--no-tune --num-env-runners=0`
which should allow you to set breakpoints anywhere in the RLlib code and
have the execution stop there for inspection and debugging.

For logging to your WandB account, use:
`--wandb-key=[your WandB API key] --wandb-project=[some project name]
--wandb-run-name=[optional: WandB run name (within the defined project)]`

"""
#from typing import Dict, Tuple
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env


parser = add_rllib_example_script_args(
    default_iters=10, #default_iters=200,
    default_timesteps=1000000,
    default_reward=300,
)
parser.add_argument(
    "--num-pols", type=int, default=0, help="pre-trained pols test",
)
parser.add_argument(
    "--path", type=str, default=None, help="absolute path to checkpoint",
)
parser.add_argument(
    "--env", type=str, default="waterworld",
    choices=["waterworld"],
    help="The environment to use."
    "`waterworld`: SISL WaterWorld"
    "`multiwalker`: SISL Multiwalker (Not tested yet)."
    "`pursuit`: SISL pursuit (Not tested yet).",
)
parser.add_argument(
    "--mode", type=str, default="train",
    choices=["train","test","eval","retrain"],
    help="The script mode to execute."
    "`train`: (Default) train from scratch."
    "`test, eval`: Test agents in other sized configurations."
    "`retrain`: Train num-agents drawn from previously trained pool.",
)



class MetricCallbacks(DefaultCallbacks):
    """Class for storing all of the custom metrics used in this script"""
    def on_train_result(self, *, algorithm, result: dict, **kwargs) -> None:
        # pylint: disable-next=used-before-assignment
        result["num_agents"] = args.num_agents
        result["num_pretrained"] = args.num_pols


if __name__ == "__main__":
    args = parser.parse_args()

    assert args.num_agents > 0, "Must set --num-agents > 0 running this script!"

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
            policies=policies,
            # Exact 1:1 mapping from AgentID to ModuleID.
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

    run_rllib_example_script_experiment(base_config, args)
