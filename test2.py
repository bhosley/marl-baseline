"""Runs env in RLlib 

`python test2.py --output /root/scrap_outputs`

"""

from glob import glob

exp_dir = glob("/root/ray_results/*")
exp_dir.sort()
exp = exp_dir[-1] # most recent experiment
#
#print(exp)
#print(  glob(exp+"/PPO_env*/*")  )
#print( glob( glob(exp+"/PPO_env*")[0] + "/checkpoint*" )  )
#
try:
    checkpoint = glob(exp+"/PPO_env*/checkpoint*")[-1] # last checkpoint
finally:
    pass

trained_pols = glob(exp + "/**/*pur*", recursive=True)
# Trained Pols is a list of the location of the policy directories
# Dir contains a class and stor pickle and module state pytorch file

#print(f"{trained_pols=}")
#print(f"{trained_pols[0]=}")


from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env
from ray.rllib.algorithms.algorithm import Algorithm



parser = add_rllib_example_script_args(
    default_iters=1, default_reward=300, default_timesteps=50000)

parser.add_argument(
    "--reps", type=int, default=1,
    help="The number of repetitions of the experiment",
)


def new_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return agent_id if agent_id in trained_pols else choice(trained_pols)


#print(f"{glob(trained_pols[0]+'/*')=}")


if __name__ == "__main__":
    args = parser.parse_args()
  

    for num_test_agents in [2]:
    #for num_test_agents in range(2,8+1):
        test_pols = {f"pursuer_{i}" for i in range(num_test_agents)}
        register_env("env_test", lambda _: ParallelPettingZooEnv(waterworld_v4.parallel_env(n_pursuers=args.num_test_agents)))


        base_config = (
            get_trainable_cls(args.algo)
            .get_default_config()
            .environment("env_test")
            .rl_module(
                model_config_dict={"vf_share_layers": True},
                rl_module_spec=MultiAgentRLModuleSpec(
                    module_specs={p: SingleAgentRLModuleSpec(
                        load_state_path=checkpoint+"/learner_group/learner/rl_module/"+p
                    ) for p in test_pols},
                ),
            )

            .multi_agent(
                policies=test_pols,
                # Exact 1:1 mapping from AgentID to ModuleID.
                #policy_mapping_fn=(lambda aid, *args, **kwargs: aid),
                policy_mapping_fn=new_policy_mapping_fn
            )

        )


        #################
        #print(f"{len(trained_pols)=}")
        exit()
        #################  

        run_rllib_example_script_experiment(base_config, args)

"""
    restored_algo = Algorithm.from_checkpoint(
        checkpoint=path_to_checkpoint,
        policy_ids=test_pols,  # <- restore only those policy IDs here.
        policy_mapping_fn=new_policy_mapping_fn,  # <- use this new mapping fn.
    )

    restored_algo.train()
    # Terminate the new algo.
    restored_algo.stop()

+ "PPO_env


"""