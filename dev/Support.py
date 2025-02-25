from abc import abstractmethod
import numpy as np
from glob import glob

import gymnasium as gym

from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env

# pylint: disable=C0415

def get_eligible_policies(args):
    """Return a list of policy objects based on parsed args"""
    recon_path = f'{args.prefix}{args.env}/{args.algo}/{args.trained_agents}_agent/'
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

def get_policy_set(pols, n, replacement=False):
    """Takes pretrained policy set, and returns a new set of n-length"""
    if replacement:
        return np.random.choice(pols,n)
    else:
        new_set = []
        while len(new_set) < n:
            dif = min(len(pols),n-len(new_set))
            for e in np.random.choice(pols, dif, replace=False):
                new_set.append(e)
        return new_set

####
def get_policies_from_checkpoint(path, n=None, replacement=False):
    pols = [Policy.from_checkpoint(p) for p in glob(f"{path}/policies/*")]
    if n:
        return get_policy_set(pols, n, replacement)
    else:
        return pols
####


class EnvironmentBase():
    """An abstract base-class to support the functions based on environment"""
    def __init__(self, **kwargs) -> None:
        self.env_name = kwargs.get('env_name', 'unnammmed_environment')
        self.agent_name = kwargs.get('agent_name', 'unnamed_agent')
        self.agent_range = kwargs.get('agent_range',range(2,5))
        self.plateau_std = kwargs.get('plateau_std',2)

    def blank_policies(self, num_agents=None) -> set:
        """Return a set of n policy names, default to max test range"""
        n = num_agents or self.agent_range.stop
        return {f"{self.agent_name}_{i}" for i in range(n)}

    def get_policies_from_checkpoint(self, path, n=None, replacement=False):
        pols = [Policy.from_checkpoint(p) for p in glob(f"{path}/policies/*")]
        if n:
            return get_policy_set(pols, n, replacement)
        else:
            return pols

    @abstractmethod
    def register(self, num_agents) -> None:
        """Register environment with tune's env registry"""


class Waterworld(EnvironmentBase):
    """Waterworld-v4 Wrapper; testing is on 2-8 agent environments"""
    from pettingzoo.sisl import waterworld_v4
    def __init__(self, n_coop=2):
        super().__init__(
            env_name = 'waterworld',
            agent_name = 'pursuer',
            agent_range = range(2,9),
        )
        self.n_coop = n_coop

    def register(self, num_agents, n_coop=2) -> None:
        register_env(f"{num_agents}_agent_{self.env_name}", lambda _:
                ParallelPettingZooEnv(
                    self.waterworld_v4.parallel_env(
                        n_pursuers=num_agents, n_coop=self.n_coop
                    )))


class Pursuit(EnvironmentBase):
    """Pursuit-v4 Wrapper; testing is on 2-8 agent environments"""
    from pettingzoo.sisl import pursuit_v4
    def __init__(self):
        super().__init__(
            env_name = 'pursuit',
            agent_name = 'pursuer',
            agent_range = range(2,9),
        )

    def register(self, num_agents) -> None:
        register_env(f"{num_agents}_agent_{self.env_name}", lambda _:
            ParallelPettingZooEnv(
                self.pursuit_v4.parallel_env(
                    n_pursuers=num_agents,
                    obs_range=10))) # Default for pursuit is 7, but the 
                    # smallest rllib supports (without another wrapper) is 10


class MultiWalker(EnvironmentBase):
    """Multiwalker-v9 Wrapper"""
    from pettingzoo.sisl import multiwalker_v9
    def __init__(self):
        super().__init__(
            env_name = 'multiwalker',
            agent_name = 'walker',
            agent_range = range(3,9), # Need to test decent top metric
        )

    def register(self, num_agents) -> None:
        register_env(f"{num_agents}_agent_{self.env_name}", lambda _:
            ParallelPettingZooEnv(
                self.multiwalker_v9.parallel_env(n_walkers=num_agents)))

    #>=python 3.12# @typing.override 
    def get_policies_from_checkpoint(self, path, n=None, replacement=False, structured=True):
        pols = [Policy.from_checkpoint(p) for p in glob(f"{path}/policies/*")]
        if structured:
            return [pols[0], *np.random.choice(pols[1:-1],n), pols[-1]]
        else:
            return super.get_policies_from_checkpoint(path, n, replacement)


class ForageWrapper():
    def __init__(self, num_agents=2, dim=8, max_food=1, coop=False):
        import lbforaging
        #self.env = gym.make("Foraging-8x8-2p-1f-v3") 
        c = "-coop" if coop else ""
        self.env = gym.make(f"Foraging-{dim}x{dim}-{num_agents}p-{max_food}f{c}-v3") 
        
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


class Foraging(EnvironmentBase):
    """Level Based Foraging v3"""
    def __init__(self):
        import lbforaging
        super().__init__(
            env_name = 'foraging',
            agent_name = 'forager',
            agent_range = range(2,10), # Between 2 and 9 agents
            plateau_std = 0.025
        )

    def register(self, **kwargs) -> None:
        # import lbforaging
        num_agents = kwargs.get("num_agents", "Unknown")
        def func():
            import lbforaging
            return ParallelPettingZooEnv(
                ForageWrapper(**kwargs)
            )

        register_env(f"{num_agents}_agent_{self.env_name}", lambda _: func())

