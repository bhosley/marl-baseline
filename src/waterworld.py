"""Runs the PettingZoo Waterworld env in RLlib using independent multi-agent learning.

See: https://pettingzoo.farama.org/environments/sisl/waterworld/
for more details on the environment.


How to run this script
----------------------
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

from pettingzoo.sisl import waterworld_v4, pursuit_v4, multiwalker_v9

from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env


parser = add_rllib_example_script_args(
    default_iters=10,#200,
    default_timesteps=1000000,
    #default_reward=0.0,
)


if __name__ == "__main__":
    args = parser.parse_args()

    assert args.num_agents > 0, "Must set --num-agents > 0 when running this script!"

    register_env("env", lambda _: PettingZooEnv(waterworld_v4.env()))
    policies = {f"pursuer_{i}" for i in range(args.num_agents)}

    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment(
            "env",
            env_config={"n_pursuers": args.num_agents},
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