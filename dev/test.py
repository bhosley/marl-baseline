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
import numpy as np
from random import shuffle
from argparse import ArgumentParser

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.utils.test_utils import (add_rllib_example_script_args,
                                        run_rllib_example_script_experiment)
from ray.tune.registry import get_trainable_cls, register_env


parser = add_rllib_example_script_args(
    ArgumentParser(conflict_handler='resolve'), # Resolve for env
    default_iters=200,
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


def get_policy_set(pols, n, rand=True):
    """Takes pretrained policy set, and returns a new set of n-length"""
    if rand:
        return np.random.choice(pols,n)
    elif n <= len(pols):
        return np.random.choice(pols,n,replace=False)
    else:
        new_set = np.append(
            np.array(pols*int(n/len(pols))),
            np.random.choice(pols,n%len(pols),replace=False)
        )
        shuffle(new_set)
        return new_set



if __name__ == "__main__":
    args = parser.parse_args()


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


    # The common-case configuration
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






    if args.mode == 'train':
        assert args.num_agents > 0, "Must set --num-agents > 0 when training!"
        run_rllib_example_script_experiment(base_config, args)
        exit()






    if args.mode != 'train':

        from glob import glob
        from os import path
        # pylint: disable=ungrouped-imports
        from ray.rllib.policy.policy import Policy

        # Loading function:
        if args.path:
            # pylint: disable=fixme
            # TODO: Check validity of provided path
            path_to_checkpoint = args.path
        else:
            raise TypeError("Missing path to checkpoint argument.")
            #print("No path provided, searching RLlib default location")
            # pylint: disable=fixme
            # TODO: need to locate and assign a path automatically

        trained_pols = glob(path_to_checkpoint+"/policies/*")
        trained_pols = {path.basename(p) : Policy.from_checkpoint(p) for p in trained_pols}
        args.num_pols = len(trained_pols)








    if args.mode in ('eval', 'test'):
        # Wandb integration
        if hasattr(args, "wandb_key") and args.wandb_key is not None:
            import wandb
            wandb.init(project = args.wandb_project or "Eval_Test")


        for test_agents in range(2,9):
            register_env(f"{test_agents}_agent_{args.env}", lambda _:
                ParallelPettingZooEnv(
                    waterworld_v4.parallel_env(n_pursuers=test_agents)))
            policies = {f"pursuer_{i}" for i in range(test_agents)}

            base_config = (
                get_trainable_cls(args.algo)
                .get_default_config()
                .environment( f"{test_agents}_agent_{args.env}" )
                .multi_agent(
                    policies=policies,
                    # Exact 1:1 mapping from AgentID to ModuleID.
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

            for _ in range(args.num_samples):
                algo = base_config.build()
                new_pols = get_policy_set([*trained_pols.values()],test_agents,False)
                for i in range(test_agents):
                    algo.get_policy(f"pursuer_{i}").set_weights(new_pols[i].get_weights())

                out = algo.evaluate()
                out["env_runners"]['test_agents'] = test_agents
                out["env_runners"]['trained_agents'] = args.num_pols
                if hasattr(args, "wandb_key") and args.wandb_key is not None:
                    wandb.log(out["env_runners"])

                algo.stop()




        #trained_pols = {f"pursuer_{i}" for i in range(trained_agents)}

    # Policy-Agent Assignment Methods
    ##resto_algo = resto_config.build()
    ##for test_id in range(num_test_agents):
    ##    train_id = np.random.randint(num_trained_agents)
    ##    resto_algo.get_policy(f"pursuer_{test_id}").set_weights(specs[f"pursuer_{train_id}"].get_weights())

    # Secondary Training:
    #   need new env number
    #if args.mode == 'retrain':
    #    pass