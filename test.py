"""Runs env in RLlib 

How to run this script
----------------------
`
python test.py --wandb-project=mini_test_ww2 --wandb-key=913528a8e92bf601b6eb055a459bcc89130c7f5f \
--num-samples=10 --num-env-runners=30 --enable-new-api-stack --checkpoint-at-end --num-agents=3
`

python test.py --output /root/scrap_outputs --num-agents=2

--wandb-run-name 



`python [script file name].py --enable-new-api-stack --num-agents=2`

Control the number of agents and policies (RLModules) via --num-agents and
--num-policies.

This works with hundreds of agents and policies, but note that initializing
many policies might take some time.

For debugging, use the following additional command line options
`--no-tune --num-env-runners=0`
which should allow you to set breakpoints anywhere in the RLlib code and
have the execution stop there for inspection and debugging.

For logging to your WandB account, use:
`--wandb-key=[your WandB API key] --wandb-project=[some project name]
--wandb-run-name=[optional: WandB run name (within the defined project)]`

"""
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env


parser = add_rllib_example_script_args(
    default_iters=200,
    default_timesteps=1000000,
    default_reward=0.0,
)

from pettingzoo.sisl import waterworld_v4


def new_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return agent_id if agent_id in trained_pols else choice(trained_pols)



if __name__ == "__main__":
    args = parser.parse_args()

    assert args.num_agents > 0, "Must set --num-agents > 0 when running this script!"
    
    #register_env("env", lambda _: PettingZooEnv(waterworld_v4.env()))
    #register_env("env", lambda _: PettingZooEnv(waterworld_v4.parallel_env(n_pursuers=args.num_agents)))
    register_env(f"{args.num_agents}_agent_env", lambda _: ParallelPettingZooEnv(waterworld_v4.parallel_env(n_pursuers=args.num_agents)))
    policies = {f"pursuer_{i}" for i in range(args.num_agents)}
        
    #register_env("env", lambda _: PettingZooEnv(multiwalker_v9.env()))
    #policies = {f"walker_{i}" for i in range(args.num_agents)}

    #register_env("env", lambda _: PettingZooEnv(pursuit_v4.env()))
    #policies = {f"pursuer_{i}" for i in range(args.num_agents)}

    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment(
            #waterworld_v4,#.env(n_pursuers=args.num_agents)
            f"{args.num_agents}_agent_env",
            #env_config={"n_pursuers": args.num_agents},
        )
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
    )

    run_rllib_example_script_experiment(base_config, args)
    #results = run_rllib_example_script_experiment(base_config, args)

    #path_to_checkpoint = results.experiment_path

    #for test_agents in range(2,8+1):
    #    test_pols = {f"pursuer_{i}" for i in range(test_agents)}
#
    #    restored_algo = Algorithm.from_checkpoint(
    #        checkpoint=path_to_checkpoint,
    #        policy_ids=test_pols,  # <- restore only those policy IDs here.
    #        policy_mapping_fn=new_policy_mapping_fn,  # <- use this new mapping fn.
    #    )