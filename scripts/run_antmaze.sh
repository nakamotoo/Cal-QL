export D4RL_SUPPRESS_IMPORT_ERROR=1
# export CUDA_VISIBLE_DEVICES=0
# export WANDB_DISABLED=True

# antmaze-medium-diverse-v2, antmaze-medium-play-v2, antmaze-large-diverse-v2, antmaze-large-play-v2
env=antmaze-medium-diverse-v2

XLA_PYTHON_CLIENT_PREALLOCATE=false python -m JaxCQL.conservative_sac_main \
    --env $env \
    --logging.online \
    --seed 0 \
    --logging.project=Cal-QL-exapmle \
    --cql_min_q_weight=5.0 \
    --cql.cql_target_action_gap=0.8 \
    --cql.cql_lagrange=True \
    --policy_arch=256-256 \
    --qf_arch=256-256-256-256 \
    --offline_eval_every_n_epoch=50 \
    --online_eval_every_n_env_steps=2000 \
    --eval_n_trajs=20 \
    --n_train_step_per_epoch_offline=1000 \
    --n_pretrain_epochs=1000 \
    --max_online_env_steps=1e6 \
    --mixing_ratio=0.5 \
    --reward_scale=10.0 \
    --reward_bias=-5 \
    --enable_calql=True 
