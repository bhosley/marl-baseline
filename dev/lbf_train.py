from argparse import ArgumentParser
from glob import glob

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.test_utils import (add_rllib_example_script_args,
                                        run_rllib_example_script_experiment)
from ray.tune.registry import get_trainable_cls
from ray.tune.stopper import (  CombinedStopper, TrialPlateauStopper,
                                MaximumIterationStopper, FunctionStopper)


class InitialCallbacks(DefaultCallbacks):
    """Class for storing all of the custom metrics used in this script"""
    def on_train_result(self, *, algorithm, result: dict, **kwargs) -> None:
        try:
            result["num_agents"] = len(result['info']['learner'])
            result["episode_reward_mean"] = (
                result['env_runners']["episode_reward_mean"])
        except:
            pass

class RetrainCallbacks(InitialCallbacks):
    """Class for storing all of the custom callbacks used in this script"""
    def on_algorithm_init(self, *, algorithm, **kwargs) -> None:
        """Callback to do the rebuilding.""" 
        from Support import get_policies_from_checkpoint

        n = len(algorithm.config.policies)
        #new_pols = get_policies_from_checkpoint(args.path, n)
        new_pols = env.get_policies_from_checkpoint(args.path, n)
        for i in range(n):
            algorithm.remove_policy(f'{env.agent_name}_{i}')
            algorithm.add_policy(f'{env.agent_name}_{i}', policy=new_pols[i])



parser = add_rllib_example_script_args(
    ArgumentParser(conflict_handler='resolve'), # Resolve for env
    default_iters=200,
    default_timesteps=1000000,
    default_reward=300,
)
parser.add_argument(
    "--env", type=str, default="waterworld",
    choices=["waterworld","multiwalker","pursuit"],
    help="The environment to use."
    "`waterworld`: SISL WaterWorld"
    "`multiwalker`: SISL Multiwalker (Not tested yet)."
    "`pursuit`: SISL pursuit (Not tested yet).",
)
parser.add_argument(
    "--retrain", action="store_true",
    help="flag for continuing training from checkpoint",
) #Perhaps redundant as retraining will require a pretrained pathway.
parser.add_argument(
    "--path", type=str, default=None, required=False,
    help="absolute path to checkpoint",
)
parser.add_argument(
    "--steps_pretrained", type=int, default=0,
    help="The number of iterations pretrained before this script."
)
parser.add_argument(
    "-r", "--replacement", action="store_true",
    help="Use replacement of elements in policy selection method.",
)

# Args for LBF
parser.add_argument(
    "--size", type=int, default=8,
    help="The dimension of the environment.",
)
parser.add_argument(
    "--coop", action='store_true',
    help="Force cooperative mode."
    "Requiring cooperation to consume regardless of level.",
)
parser.add_argument(
    "--food", type=int, default=1,
    help="Set the max food parameter.",
)



if __name__ == "__main__":
    args = parser.parse_args()
    assert args.num_agents > 0, "Must set --num-agents > 0 when training!"

    from Support import Foraging   
    args.env = "foraging"
    env = Foraging()
    env.register(num_agents=args.num_agents, max_food=args.food)
    policies = env.blank_policies(args.num_agents)

    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment( f"{args.num_agents}_agent_{args.env}", )
        .multi_agent(
            policies=policies, # Exact 1:1 mapping from AgentID to ModuleID.
            policy_mapping_fn=(lambda aid, *args, **kwargs: aid),
        )
        .rl_module(
            model_config_dict={"vf_share_layers": True,},
            rl_module_spec=MultiRLModuleSpec( 
                rl_module_specs={p: RLModuleSpec() for p in policies},
            ),
        )
        
    )

    

    # Reimplement stopping criteria; including original max iters,
    # max timesteps, max reward, and adding plateau
    stopper = CombinedStopper(
        MaximumIterationStopper(max_iter=args.stop_iters),
        FunctionStopper(lambda trial_id, result: (
            result["num_env_steps_sampled_lifetime"] >= args.stop_timesteps)
        ),
        FunctionStopper(lambda trial_id, result: (
            result["episode_reward_mean"] >= args.stop_reward)
        ),
        TrialPlateauStopper(
            metric="episode_reward_mean",
            num_results=15, std=env.plateau_std
        ),
    )
    
    if not args.path:
        # This is treated as initial training:
        config = base_config.callbacks(InitialCallbacks)
        config["steps_pretrained"] = 0
    else:
        # This is treated as retraining:
        config = base_config.callbacks(RetrainCallbacks)
        config["steps_pretrained"] = args.steps_pretrained  
        # Is it possible to get pretrained steps from restore?
        config["num_pretrained_agents"] = len(glob(f"{args.path}/policies/*"))

    # Record information
    config["num_agents"] = args.num_agents

    # Conduct the experiment
    ress = run_rllib_example_script_experiment(config, args, stop=stopper)

    # Organize results
    if args.checkpoint_at_end:
        import shutil
        for i, res in enumerate(ress):
            source = res.path + "/checkpoint_000000"
            score = res.metrics['env_runners']['episode_reward_mean']
            dest = (f"/root/test/{args.env}/{args.algo}/" +
                    f"{args.num_agents}_agent/{score}")
            shutil.copytree(source, dest, dirs_exist_ok=True)

    for res in ress: 
        print(res.path)
    exit()

"""
python lbf_train.py --num-samples=2 --num-env-runners=30 --num-agents=2 --stop-iters=10 --checkpoint-freq=2

tmux new-session -d "python lbf_train.py --num-samples=10 --num-env-runners=30 --num-agents=2 --wandb-key=913528a8e92bf601b6eb055a459bcc89130c7f5f --wandb-project=foraging-test  --stop-iters=500 --stop-timesteps=100000000"


"""