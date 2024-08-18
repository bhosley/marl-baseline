# Output

## Step()

```txt
{
    'env_runners': {
        'agent_episode_returns_mean': {
            'pursuer_0': 34.04282704821171, 
            'pursuer_1': -38.5849330860362
        }, 
        'agent_steps': {
            'pursuer_0': 500.0, 
            'pursuer_1': 500.0
        }, 
        'episode_duration_sec_mean': 1.885190265290439, 
        'episode_len_max': 1000, 
        'episode_len_mean': 1000.0, 
        'episode_len_min': 1000, 
        'episode_return_max': 160.3325663997169, 
        'episode_return_mean': -4.542106037824476, 
        'episode_return_min': -307.300210613529, 
        'module_episode_returns_mean': {
            'pursuer_0': 34.04282704821171, 
            'pursuer_1': -38.5849330860362
        }, 
        'num_agent_steps_sampled': {
            'pursuer_0': 2000, 
            'pursuer_1': 2004
        }, 
        'num_agent_steps_sampled_lifetime': {
            'pursuer_0': 40202000, 
            'pursuer_1': 40282404
        }, 
        'num_env_steps_sampled': 4000, 
        'num_env_steps_sampled_lifetime': 160004000, 
        'num_episodes': 4, 
        'num_module_steps_sampled': {
            'pursuer_0': 2000, 
            'pursuer_1': 2004
        }, 
        'num_module_steps_sampled_lifetime': {
            'pursuer_0': 40202000, 
            'pursuer_1': 40282404
        }
    }, 
    'learners': {
        '__all_modules__': {
            'num_env_steps_trained': 4000, '
            num_module_steps_trained': 4000, 
            'num_non_trainable_parameters': 0.0, 
            'num_trainable_parameters': 517140.0, 
            'total_loss': -1.4884347915649414
        }, 
        'pursuer_0': {
            'curr_entropy_coeff': 0.0, 
            'curr_kl_coeff': 0.30000001192092896, 
            'default_optimizer_learning_rate': 5e-05, 
            'entropy': 2.890568971633911, 
            'mean_kl_loss': 0.06395142525434494, 
            'num_module_steps_trained': 2000, 
            'num_non_trainable_parameters': 0.0, 
            'num_trainable_parameters': 129285.0, 
            'policy_loss': -0.42700645327568054, 
            'total_loss': -1.4884347915649414, 
            'vf_explained_var': 0.024785280227661133, 
            'vf_loss': 7.720778465270996, 
            'vf_loss_unclipped': 112.71151733398438
        }, 
        'pursuer_1': {
            'curr_entropy_coeff': 0.0, 
            'curr_kl_coeff': 0.30000001192092896, 
            'default_optimizer_learning_rate': 5e-05, 
            'entropy': 2.5336380004882812, 
            'mean_kl_loss': 0.20875799655914307, 
            'num_module_steps_trained': 2000, 
            'num_non_trainable_parameters': 0.0, 
            'num_trainable_parameters': 129285.0, 
            'policy_loss': -1.1980079412460327, 
            'total_loss': -1.1128225326538086, 
            'vf_explained_var': -0.04609668254852295, 
            'vf_loss': 8.686759948730469, 
            'vf_loss_unclipped': 304.0238037109375
        }, 
        'pursuer_2': {
            'curr_entropy_coeff': 0.0, 
            'default_optimizer_learning_rate': 5e-05, 
            'num_non_trainable_parameters': 0.0, 
            'num_trainable_parameters': 129285.0
        }, 
        'pursuer_3': {
            'curr_entropy_coeff': 0.0, 
            'default_optimizer_learning_rate': 5e-05, 
            'num_non_trainable_parameters': 0.0, 
            'num_trainable_parameters': 129285.0
        }
    }, 
    'num_agent_steps_sampled_lifetime': {
        'pursuer_0': 402000, 
        'pursuer_1': 402804
    }, 
    'num_env_steps_sampled_lifetime': 804000, 
    'num_env_steps_trained_lifetime': 804000, 
    'num_episodes_lifetime': 804, 
    'timers': {
        'env_runner_sampling_timer': 3.8473340517118517, 
        'learner_update_timer': 4.389978385980097, 
        'synch_env_connectors': 0.007852348634711346, 
        'synch_weights': 0.012520314034742892
    }, 
    'fault_tolerance': {
        'num_healthy_workers': 2, 
        'num_in_flight_async_reqs': 0, 
        'num_remote_worker_restarts': 0
    }
}
```

## Train()

```txt
{
    'env_runners': {
        'agent_episode_returns_mean': {
            'pursuer_0': 26.615716641695506, 
            'pursuer_1': -50.73190415936184
        }, 
        'agent_steps': {
            'pursuer_0': 500.0, 
            'pursuer_1': 500.0
        }, 
        'episode_duration_sec_mean': 1.947082866737619, 
        'episode_len_max': 1000, 
        'episode_len_mean': 1000.0, 
        'episode_len_min': 1000, 
        'episode_return_max': 160.3325663997169, 
        'episode_return_mean': -24.116187517666297, 
        'episode_return_min': -307.300210613529, 
        'module_episode_returns_mean': {
            'pursuer_0': 26.615716641695506, 
            'pursuer_1': -50.73190415936184
        }, 
        'num_agent_steps_sampled': {
            'pursuer_0': 2000, 
            'pursuer_1': 2004
        }, 
        'num_agent_steps_sampled_lifetime': {
            'pursuer_0': 40206000, 
            'pursuer_1': 40286412
        }, 
        'num_env_steps_sampled': 4000, 
        'num_env_steps_sampled_lifetime': 161616000, 
        'num_episodes': 4, 
        'num_module_steps_sampled': {
            'pursuer_0': 2000, 
            'pursuer_1': 2004
        }, 
        'num_module_steps_sampled_lifetime': {
            'pursuer_0': 40206000, 
            'pursuer_1': 40286412
        }
    }, 
    'learners': {
        '__all_modules__': {
            'num_env_steps_trained': 4000, 
            'num_module_steps_trained': 4000, 
            'num_non_trainable_parameters': 0.0, 
            'num_trainable_parameters': 517140.0, 
            'total_loss': -2.6306087970733643
        }, 
        'pursuer_0': {
            'curr_entropy_coeff': 0.0, 
            'curr_kl_coeff': 0.30000001192092896, 
            'default_optimizer_learning_rate': 5e-05, 
            'entropy': 2.7935426235198975, 
            'mean_kl_loss': 0.0057118134573102, 
            'num_module_steps_trained': 2000, 
            'num_non_trainable_parameters': 0.0, 
            'num_trainable_parameters': 129285.0, 
            'policy_loss': -0.6233325004577637, 
            'total_loss': -2.6306087970733643, 
            'vf_explained_var': -0.018345355987548828, 
            'vf_loss': 10.0, 'vf_loss_unclipped': 158.31146240234375
        }, 
        'pursuer_1': {
            'curr_entropy_coeff': 0.0, 
            'curr_kl_coeff': 0.30000001192092896, 
            'default_optimizer_learning_rate': 5e-05, 
            'entropy': 2.291994571685791, 
            'mean_kl_loss': 0.016567206010222435, 
            'num_module_steps_trained': 2000, 
            'num_non_trainable_parameters': 0.0, 
            'num_trainable_parameters': 129285.0, 
            'policy_loss': -2.1113224029541016, 
            'total_loss': -2.0589897632598877, 
            'vf_explained_var': 0.06778156757354736, 
            'vf_loss': 9.47250747680664, 
            'vf_loss_unclipped': 237.19288635253906
        }, 
        'pursuer_2': {
            'curr_entropy_coeff': 0.0, 
            'default_optimizer_learning_rate': 5e-05, 
            'num_non_trainable_parameters': 0.0, 
            'num_trainable_parameters': 129285.0
        }, 
        'pursuer_3': {
            'curr_entropy_coeff': 0.0, 
            'default_optimizer_learning_rate': 5e-05, 
            'num_non_trainable_parameters': 0.0, 
            'num_trainable_parameters': 129285.0
        }
    }, 
    'num_agent_steps_sampled_lifetime': {
        'pursuer_0': 404000, 
        'pursuer_1': 404808
    }, 
    'num_env_steps_sampled_lifetime': 808000, 
    'num_env_steps_trained_lifetime': 808000, 
    'num_episodes_lifetime': 808, 
    'timers': {'env_runner_sampling_timer': 3.8765751347506137, 
    'learner_update_timer': 13.514154431752878, 
    'synch_env_connectors': 0.008218620302399324, 
    'synch_weights': 0.012538271099729888}, 
    'fault_tolerance': {
        'num_healthy_workers': 2, 
        'num_in_flight_async_reqs': 0, 
        'num_remote_worker_restarts': 0
    }, 
    'done': False, 
    'training_iteration': 201, 
    'trial_id': 'default', 
    'date': '2024-08-18_20-03-50', 
    'timestamp': 1724011430, 
    'time_this_iter_s': 923.6468715667725, 
    'time_total_s': 923.6468715667725, 
    'pid': 888287, 
    'hostname': 'e-bgbfbjbn-z5drr-0',
    'node_ip': '10.42.142.236', 
    'config': {
        'extra_python_environs_for_driver': {}, 
        'extra_python_environs_for_worker': {}, 
        'placement_strategy': 'PACK', 
        'num_gpus': 0, 
        '_fake_gpus': False, 
        'eager_tracing': True, 
        'eager_max_retraces': 20, 
        'tf_session_args': {
            'intra_op_parallelism_threads': 2, 
            'inter_op_parallelism_threads': 2, 
            'gpu_options': {
                'allow_growth': True
            }, 
            'log_device_placement': False, 
            'device_count': {
                'CPU': 1
            }, 
            'allow_soft_placement': True
        }, 
        'local_tf_session_args': {
            'intra_op_parallelism_threads': 8, 
            'inter_op_parallelism_threads': 8
        }, 
    'torch_compile_learner': False, 
    'torch_compile_learner_what_to_compile': <TorchCompileWhatToCompile.FORWARD_TRAIN: 'forward_train'>, 
    'torch_compile_learner_dynamo_backend': 'inductor', 
    'torch_compile_learner_dynamo_mode': None, 
    'torch_compile_worker': False, 
    'torch_compile_worker_dynamo_backend': 'onnxrt', 
    'torch_compile_worker_dynamo_mode': None, 
    'enable_rl_module_and_learner': True, 
    'enable_env_runner_and_connector_v2': True, 
    'env': 'env', 
    'env_config': {}, 
    'observation_space': None, 
    'action_space': None, 
    'clip_rewards': None, 
    'normalize_actions': True, 
    'clip_actions': False, 
    '_is_atari': None, 
    'env_task_fn': None, 
    'render_env': False, 
    'action_mask_key': 'action_mask', 
    'env_runner_cls': None, 
    'num_envs_per_env_runner': 1, 
    'custom_resources_per_env_runner': {}, 
    'validate_env_runners_after_construction': True, 
    'sample_timeout_s': 60.0, 
    '_env_to_module_connector': None, 
    'add_default_connectors_to_env_to_module_pipeline': True, 
    '_module_to_env_connector': None, 
    'add_default_connectors_to_module_to_env_pipeline': True, 
    'episode_lookback_horizon': 1, 
    'rollout_fragment_length': 'auto', 
    'batch_mode': 'truncate_episodes', 
    'compress_observations': False, 
    'remote_worker_envs': False, 
    'remote_env_batch_wait_ms': 0, 
    'enable_tf1_exec_eagerly': False, 
    'sample_collector': <class 'ray.rllib.evaluation.collectors.simple_list_collector.SimpleListCollector'>, 
    'preprocessor_pref': 'deepmind', 
    'observation_filter': 'NoFilter', 
    'update_worker_filter_stats': True, 
    'use_worker_filter_stats': True, 
    'enable_connectors': True, 
    'sampler_perf_stats_ema_coef': None, 
    'local_gpu_idx': 0, 
    'gamma': 0.99, 
    'lr': 5e-05, 
    'grad_clip': None, 
    'grad_clip_by': 
    'global_norm', 
    'train_batch_size': 4000, 
    'train_batch_size_per_learner': None, 
    'model': {
        '_disable_preprocessor_api': False, 
        '_disable_action_flattening': False, 
        'fcnet_hiddens': [256, 256], 
        'fcnet_activation': 'tanh', 
        'fcnet_weights_initializer': None, 
        'fcnet_weights_initializer_config': None, 
        'fcnet_bias_initializer': None, 
        'fcnet_bias_initializer_config': None, 
        'conv_filters': None, 'conv_activation': 'relu', 
        'conv_kernel_initializer': None, 
        'conv_kernel_initializer_config': None, 
        'conv_bias_initializer': None, 
        'conv_bias_initializer_config': None, 
        'conv_transpose_kernel_initializer': None, 
        'conv_transpose_kernel_initializer_config': None, 
        'conv_transpose_bias_initializer': None, 
        'conv_transpose_bias_initializer_config': None, 
        'post_fcnet_hiddens': [], 
        'post_fcnet_activation': 'relu', 
        'post_fcnet_weights_initializer': None, 
        'post_fcnet_weights_initializer_config': None, 
        'post_fcnet_bias_initializer': None, 
        'post_fcnet_bias_initializer_config': None, 
        'free_log_std': False, 
        'no_final_linear': False, 
        'vf_share_layers': False, 
        'use_lstm': False, 
        'max_seq_len': 20,
        'lstm_cell_size': 256, 
        'lstm_use_prev_action': False, 
        'lstm_use_prev_reward': False, 
        'lstm_weights_initializer': None, 
        'lstm_weights_initializer_config': None, 
        'lstm_bias_initializer': None, 
        'lstm_bias_initializer_config': None, 
        '_time_major': False, 'use_attention': False, 
        'attention_num_transformer_units': 1, 
        'attention_dim': 64, 
        'attention_num_heads': 1, 
        'attention_head_dim': 32, 
        'attention_memory_inference': 50, 
        'attention_memory_training': 50, 
        'attention_position_wise_mlp_dim': 32, 
        'attention_init_gru_gate_bias': 2.0, 
        'attention_use_n_prev_actions': 0, 
        'attention_use_n_prev_rewards': 0, 
        'framestack': True, 
        'dim': 84, 
        'grayscale': False, 
        'zero_mean': True, 
        'custom_model': None, 
        'custom_model_config': {}, 
        'custom_action_dist': None, 
        'custom_preprocessor': None, 
        'encoder_latent_dim': None, 
        'always_check_shapes': False, 
        'lstm_use_prev_action_reward': -1, 
        '_use_default_native_models': -1}, 
        '_learner_connector': None, 
        'add_default_connectors_to_learner_pipeline': True, 
        'optimizer': {}, 
        'max_requests_in_flight_per_sampler_worker': 2, 
        '_learner_class': None, 
        'explore': True, 
        'exploration_config': {}, 
        'algorithm_config_overrides_per_module': {}, 
        '_per_module_overrides': {}, 
        'count_steps_by': 'env_steps', 
        'policies': {
            'pursuer_1', 
            'pursuer_3', 
            'pursuer_2', 
            'pursuer_0'
        }, 
        'policy_map_capacity': 100, 
        'policy_mapping_fn': <function <lambda> at 0x7fd8f7cd72e0>, 
        'policies_to_train': None, 
        'policy_states_are_swappable': False, 
        'observation_fn': None, 
        'input_config': {}, 
        'actions_in_input_normalized': False, 
        'postprocess_inputs': False, 
        'shuffle_buffer_size': 0, 
        'output': None, 
        'output_config': {}, 
        'output_compress_columns': ['obs', 'new_obs'], 
        'output_max_file_size': 67108864, 
        'offline_sampling': False, 
        'evaluation_interval': None, 
        'evaluation_duration': 10, 
        'evaluation_duration_unit': 'episodes', 
        'evaluation_sample_timeout_s': 120.0, 
        'evaluation_parallel_to_training': False, 
        'evaluation_force_reset_envs_before_iteration': True, 
        'evaluation_config': None, 
        'off_policy_estimation_methods': {}, 
        'ope_split_batch_by_episode': True, 
        'evaluation_num_env_runners': 0, 
        'in_evaluation': False, 
        'sync_filters_on_rollout_workers_timeout_s': 10.0, 
        'keep_per_episode_custom_metrics': False, 
        'metrics_episode_collection_timeout_s': 60.0,
        'metrics_num_episodes_for_smoothing': 100, 
        'min_time_s_per_iteration': None, 
        'min_train_timesteps_per_iteration': 0, 
        'min_sample_timesteps_per_iteration': 0, 
        'export_native_model_files': False, 
        'checkpoint_trainable_policies_only': False, 
        'logger_creator': None, 
        'logger_config': None, 
        'log_level': 'WARN', 
        'log_sys_usage': True, 
        'fake_sampler': False, 
        'seed': None, 
        '_run_training_always_in_thread': False, 
        '_evaluation_parallel_to_training_wo_thread': False, 
        'ignore_env_runner_failures': False, 
        'recreate_failed_env_runners': False, 
        'max_num_env_runner_restarts': 1000, 
        'delay_between_env_runner_restarts_s': 60.0, 
        'restart_failed_sub_environments': False, 
        'num_consecutive_env_runner_failures_tolerance': 100, 
        'env_runner_health_probe_timeout_s': 30, 
        'env_runner_restore_timeout_s': 1800, 
        '_model_config_dict': {
            'vf_share_layers': True
        }, 
        '_rl_module_spec': MultiAgentRLModuleSpec(
            marl_module_class=<class 'ray.rllib.core.rl_module.marl_module.MultiAgentRLModule'>, 
            module_specs={
                'pursuer_3': SingleAgentRLModuleSpec(
                    module_class=None, observation_space=None, action_space=None, model_config_dict=None, catalog_class=None, load_state_path=None), 
                'pursuer_1': SingleAgentRLModuleSpec(
                    module_class=None, observation_space=None, action_space=None, model_config_dict=None, catalog_class=None, load_state_path=None), 
                'pursuer_2': SingleAgentRLModuleSpec(
                    module_class=None, observation_space=None, action_space=None, model_config_dict=None, catalog_class=None, load_state_path=None), 
                'pursuer_0': SingleAgentRLModuleSpec(
                    module_class=None, observation_space=None, action_space=None, model_config_dict=None, catalog_class=None, load_state_path=None)
            }, 
            load_state_path=None, 
            modules_to_load=None
        ), 
        '_AlgorithmConfig__prior_exploration_config': {'type': 'StochasticSampling'}, 
        '_tf_policy_handles_more_than_one_loss': False, 
        '_disable_preprocessor_api': False, 
        '_disable_action_flattening': False, 
        '_disable_initialize_loss_from_dummy_batch': False, 
        '_dont_auto_sync_env_runner_states': False, 
        'simple_optimizer': True, 
        'policy_map_cache': -1, 
        'worker_cls': -1, 
        'synchronize_filters': -1, 
        'enable_async_evaluation': -1, 
        'custom_async_evaluation_function': -1, 
        '_enable_rl_module_api': -1, 
        'auto_wrap_old_gym_envs': -1, 
        'disable_env_checking': -1, 
        'always_attach_evaluation_results': -1, 
        'replay_sequence_length': None, 
        '_disable_execution_plan_api': -1, 
        'lr_schedule': None, 
        'use_critic': True, 
        'use_gae': True, 
        'use_kl_loss': True, 
        'kl_coeff': 0.2, 
        'kl_target': 0.01, 
        'sgd_minibatch_size': 128, 
        'mini_batch_size_per_learner': None, 
        'num_sgd_iter': 30, 
        'shuffle_sequences': True, 
        'vf_loss_coeff': 0.005, 
        'entropy_coeff': 0.0, 
        'entropy_coeff_schedule': None, 
        'clip_param': 0.3, 
        'vf_clip_param': 10.0, 
        'vf_share_layers': -1, 
        '__stdout_file__': None, 
        '__stderr_file__': None, 
        'lambda': 1.0, 'input': 'sampler', 
        'callbacks': <class 'ray.rllib.algorithms.callbacks.DefaultCallbacks'>, 
        'create_env_on_driver': False, 
        'custom_eval_function': None, 
        'framework': 'torch', 
        'num_cpus_for_driver': 1, 
        'num_workers': 2, 
        'num_cpus_per_worker': 1, 
        'num_gpus_per_worker': 0, 
        'num_learner_workers': 0, 
        'num_cpus_per_learner_worker': 1, 
        'num_gpus_per_learner_worker': 0
    }, 
    'time_since_restore': 923.6468715667725, 
    'iterations_since_restore': 1, 
    'perf': {
        'cpu_util_percent': 30.475919559176752, 
        'ram_util_percent': 8.033476192718881
    }
}
```