export D4RL_SUPPRESS_IMPORT_ERROR=1
# export CUDA_VISIBLE_DEVICES=0
# export WANDB_DISABLED=True

env=pen-binary-v0
# env=door-binary-v0
# env=relocate-binary-v0

if [ "$env" = "pen-binary-v0" ]; then
    max_online_env_steps=2e5
elif [ "$env" = "door-binary-v0" ] || [ "$env" = "relocate-binary-v0" ]; then
    max_online_env_steps=1e6
fi

XLA_PYTHON_CLIENT_PREALLOCATE=false python -m JaxCQL.conservative_sac_main \
    --env=$env \
    --logging.online \
    --seed=0 \
    --logging.project=Cal-QL-exapmle \
    --cql_min_q_weight=1.0 \
    --policy_arch=512-512 \
    --qf_arch=512-512-512 \
    --offline_eval_every_n_epoch=2 \
    --online_eval_every_n_env_steps=2000 \
    --eval_n_trajs=20 \
    --n_train_step_per_epoch_offline=1000 \
    --n_pretrain_epochs=20 \
    --max_online_env_steps=$max_online_env_steps \
    --mixing_ratio=0.5 \
    --reward_scale=10.0 \
    --reward_bias=5.0 \
    --enable_calql=True