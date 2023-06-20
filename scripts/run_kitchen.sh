export D4RL_SUPPRESS_IMPORT_ERROR=1
# export CUDA_VISIBLE_DEVICES=0
# export WANDB_DISABLED=True

# kitchen-mixed-v0, kitchen-partial-v0, kitchen-complete-v0
env=kitchen-mixed-v0

XLA_PYTHON_CLIENT_PREALLOCATE=false python -m JaxCQL.conservative_sac_main \
    --env=$env \
    --logging.online \
    --seed=0 \
    --logging.project=Cal-QL-exapmle \
    --cql_min_q_weight=5.0 \
    --cql.cql_importance_sample=False \
    --policy_arch=512-512-512 \
    --qf_arch=512-512-512 \
    --offline_eval_every_n_epoch=50 \
    --online_eval_every_n_env_steps=5000 \
    --eval_n_trajs=20 \
    --n_train_step_per_epoch_offline=1000 \
    --n_pretrain_epochs=500 \
    --max_online_env_steps=1.5e6 \
    --mixing_ratio=0.25 \
    --enable_calql=True
