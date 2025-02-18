from argparse import ArgumentParser

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.test_utils import (add_rllib_example_script_args,
                                        run_rllib_example_script_experiment)
from ray.tune.registry import get_trainable_cls
from ray.tune.stopper import (  CombinedStopper, TrialPlateauStopper,
                                MaximumIterationStopper, FunctionStopper)






import gymnasium as gym    
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

#from gym.envs.registration import register

import gymnasium as gym

class ForageWrapper():
    def __init__(self, num_agents=2, dim=8):
        import lbforaging
        #self.env = gym.make("Foraging-8x8-2p-1f-v3") 
        self.env = gym.make(f"Foraging-{dim}x{dim}-{num_agents}p-1f-v3") 
        
        self.agents = [f"forager_{i}" for i in range(num_agents)]
        self.num_agents = num_agents
        self.max_num_agents = 9
        self.possible_agents = 9
        self.render_mode = self.env.render_mode
        self.state = None
        
    def action_spaces(self):
        return dict(zip(self.agents, self.env.action_space))

    def action_space(self, agent):
        idx = self.agents.index(agent)
        return self.env.action_space[idx]

    def observation_spaces(self):
        return dict(zip(self.agents,self.env.observation_space))
    
    def observation_space(self, agent):
        idx = self.agents.index(agent)
        return self.env.observation_space[idx]

    def reset(self, **kwargs):
        acts, infos = self.env.reset(**kwargs)
        return dict(zip(self.agents,acts)), infos

    def step(self, actions):
        acts = actions.values()
        obs, rew, term, trunc, info = self.env.step(acts)
        self.state = dict(zip(self.agents,obs))
        rews = dict(zip(self.agents,rew))
        terms = {a: term for a in self.agents}
        truncs = {a: trunc for a in self.agents}
        return self.state, rews, terms, truncs, info

    # Direct Pass
    def render(self): return self.env.render()
    def close(self): return self.env.close()
    def unwrapped(self): return self.env.unwrapped()
    def metadata(self): return self.env.metadata()



def func():
    import lbforaging
    return ParallelPettingZooEnv(
        #gym.make( "Foraging-8x8-2p-1f-v3" )
        #gym.make( "Foraging-5x5-2p-1f-v3" )
        ForageWrapper()
    )

register_env("lbf_env", lambda _: func())


class MetricCallbacks(DefaultCallbacks):
    """Class for storing all of the custom metrics used in this script"""
    def on_train_result(self, *, algorithm, result: dict, **kwargs) -> None:
        try:
            result["num_agents"] = len(result['info']['learner'])
            result["episode_reward_mean"] = (
                result['env_runners']["episode_reward_mean"])
        except:
            pass

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

if __name__ == "__main__":
    args = parser.parse_args()
    assert args.num_agents > 0, "Must set --num-agents > 0 when training!"

    args.env = "foraging"
    
    from Support import Foraging   
    env = Foraging()
    policies = env.blank_policies(args.num_agents)

    # env.register(args.num_agents)

    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        #.environment( f"{args.num_agents}_agent_{args.env}",
        .environment( "lbf_env",
        )
        .multi_agent(
            policies=policies, # Exact 1:1 mapping from AgentID to ModuleID.
            policy_mapping_fn=(lambda aid, *args, **kwargs: aid),
        )
        .rl_module(
            model_config_dict={
                "vf_share_layers": True,
                },
            rl_module_spec=MultiRLModuleSpec( 
                rl_module_specs={p: RLModuleSpec() for p in policies},
            ),
        )
        .callbacks(MetricCallbacks)
        #.training
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

    # Record information
    base_config["num_agents"] = args.num_agents
    base_config["steps_pretrained"] = 0

    # Conduct the experiment
    ress = run_rllib_example_script_experiment(base_config, args, stop=stopper)

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