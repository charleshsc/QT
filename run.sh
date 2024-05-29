#!/bin/bash

python experiment.py --seed 123 \
    --env halfcheetah --dataset medium-expert --embed_dim 256 \
    --learning_rate 3e-4 --eta 0.4 --grad_norm 15.0 \
    --exp_name qt --save_path ./save/ --batch_size 256 \
    --max_iters 10 --num_steps_per_iter 10000 --lr_decay \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 --early_epoch 10  \


python experiment.py --seed 123 \
    --env hopper --dataset medium-expert --embed_dim 256 \
    --learning_rate 3e-4 --eta 1.0 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/ --batch_size 256 \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 \

python experiment.py --seed 123 \
    --env walker2d --dataset medium-expert --embed_dim 256 \
    --learning_rate 3e-4 --eta 2.0 --grad_norm 5.0 \
    --exp_name qt --save_path ./save/ --batch_size 256 \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 \

python experiment.py --seed 123 \
    --env halfcheetah --dataset medium --embed_dim 256 \
    --learning_rate 3e-4 --eta 5.0 --grad_norm 15.0 \
    --exp_name qt --save_path ./save/ --batch_size 256 \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 --K 5 \

python experiment.py --seed 123 \
    --env hopper --dataset medium --embed_dim 256 \
    --learning_rate 3e-4 --eta 1.0 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/ --batch_size 256 \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 \

python experiment.py --seed 123 \
    --env walker2d --dataset medium --embed_dim 256 \
    --learning_rate 3e-4 --eta 2.0 --grad_norm 5.0 \
    --exp_name qt --save_path ./save/ --batch_size 256 \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 \

python experiment.py --seed 123 \
    --env halfcheetah --dataset medium-replay --embed_dim 256 \
    --learning_rate 3e-4 --eta 5.0 --grad_norm 15.0 \
    --exp_name qt --save_path ./save/ --batch_size 256 \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 --K 5 \

python experiment.py --seed 123 \
    --env hopper --dataset medium-replay --embed_dim 256 \
    --learning_rate 3e-4 --eta 3.0 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/ --batch_size 256 \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 \

python experiment.py --seed 123 \
    --env walker2d --dataset medium-replay --embed_dim 256 \
    --learning_rate 3e-4 --eta 2.0 --grad_norm 5.0 \
    --exp_name qt --save_path ./save/ --batch_size 256 \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 \

python experiment.py --seed 123 \
    --env pen --dataset human --embed_dim 256 \
    --learning_rate 3e-4 --eta 0.1 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/ --batch_size 256 \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 \

python experiment.py --seed 123 \
    --env pen --dataset cloned --embed_dim 256 \
    --learning_rate 3e-4 --eta 0.1 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/ --batch_size 256 \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 \

python experiment.py --seed 123 \
    --env hammer --dataset human --embed_dim 256 \
    --learning_rate 3e-4 --eta 0.1 --grad_norm 5.0 \
    --exp_name qt --save_path ./save/ --batch_size 256 \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 --early_epoch 60 \

python experiment.py --seed 123 \
    --env hammer --dataset cloned --embed_dim 256 \
    --learning_rate 3e-4 --eta 0.01 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/ --batch_size 256 \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 --early_epoch 30 \

python experiment.py --seed 123 \
    --env door --dataset human --embed_dim 256 \
    --learning_rate 3e-4 --eta 0.005 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/ --batch_size 256 \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 --early_epoch 60 \

python experiment.py --seed 123 \
    --env door --dataset cloned --embed_dim 256 \
    --learning_rate 3e-4 --eta 0.001 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/ --batch_size 256 \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 --early_epoch 60 \

python experiment.py --seed 312\
    --env kitchen --dataset complete --embed_dim 256 \
    --learning_rate 3e-4 --eta 0.001 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/ --batch_size 256 \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 --early_epoch 100 \

python experiment.py --seed 123 \
    --env kitchen --dataset partial --embed_dim 256 \
    --learning_rate 3e-4 --eta 0.01 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/ --batch_size 256 \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 --early_epoch 50 \

python experiment.py --seed 123 \
    --env maze2d --dataset open --embed_dim 256 \
    --learning_rate 3e-4 --eta 0.01 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/ --batch_size 256 \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 --early_epoch 50 \

python experiment.py --seed 123 \
    --env maze2d --dataset open-dense --embed_dim 256 \
    --learning_rate 3e-4 --eta 0.01 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/ --batch_size 256 \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 --early_epoch 50 \

python experiment.py --seed 123 \
    --env maze2d --dataset umaze --embed_dim 256 \
    --learning_rate 3e-4 --eta 5.0 --grad_norm 20.0 \
    --exp_name qdt_v2 --save_path ./save/ --batch_size 256 \
    --max_iters 100 --num_steps_per_iter 1000 --lr_decay \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 --early_epoch 50 \

python experiment.py --seed 123 \
    --env maze2d --dataset umaze-dense --embed_dim 256 \
    --learning_rate 3e-4 --eta 3.0 --grad_norm 5.0 \
    --exp_name qdt_v2 --save_path ./save/ --batch_size 256 \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 --early_epoch 50 \

python experiment.py --seed 123 \
    --env maze2d --dataset medium --embed_dim 256 \
    --learning_rate 3e-4 --eta 5.0 --grad_norm 9.0 \
    --exp_name qdt_v2 --save_path ./save/ --batch_size 256 \
    --max_iters 100 --num_steps_per_iter 1000 --lr_decay \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 --early_epoch 50 \

python experiment.py --seed 123 \
    --env maze2d --dataset medium-dense --embed_dim 256 \
    --learning_rate 3e-4 --eta 5.0 --grad_norm 9.0 \
    --exp_name qdt_v2 --save_path ./save/ --batch_size 256 \
    --max_iters 100 --num_steps_per_iter 1000 --lr_decay \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 --early_epoch 50 \

python experiment.py --seed 123 \
    --env maze2d --dataset large --embed_dim 256 \
    --learning_rate 3e-4 --eta 4.0 --grad_norm 9.0 \
    --exp_name qdt_v2 --save_path ./save/ --batch_size 256 \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 --early_epoch 50 \

python experiment.py --seed 123 \
    --env maze2d --dataset large-dense --embed_dim 256 \
    --learning_rate 3e-4 --eta 4.0 --grad_norm 9.0 \
    --exp_name qdt_v2 --save_path ./save/ --batch_size 256 \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 --early_epoch 50 \

python experiment.py --seed 123 \
    --env antmaze --dataset umaze --embed_dim 256 \
    --learning_rate 3e-4 --eta 0.05 --grad_norm 9.0 \
    --exp_name qdt_v2 --save_path ./save/ --batch_size 256 \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 --early_epoch 50 \

python experiment.py --seed 123 \
    --env antmaze --dataset umaze-diverse --embed_dim 256 \
    --learning_rate 3e-4 --eta 0.01 --grad_norm 9.0 \
    --exp_name qdt_v2 --save_path ./save/ --batch_size 256 \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 --early_epoch 50 \

python experiment.py --seed 123 \
    --env antmaze --dataset medium-diverse --embed_dim 256 \
    --learning_rate 3e-4 --eta 0.01 --grad_norm 9.0 \
    --exp_name qdt_v2 --save_path ./save/ --batch_size 256 \
    --max_iters 100 --num_steps_per_iter 1000 --lr_decay --num_eval_episodes 10 \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 --early_epoch 80 \

python experiment.py --seed 123 \
    --env antmaze --dataset large-diverse --embed_dim 256 \
    --learning_rate 3e-4 --eta 0.005 --grad_norm 9.0 \
    --exp_name qdt_v2 --save_path ./save/ --batch_size 256 \
    --max_iters 100 --num_steps_per_iter 1000 --lr_decay --num_eval_episodes 10 \
    --n_layer 4 --n_head 4  --early_stop --k_rewards --use_discount --discount 0.99 --early_epoch 80 --reward_tune cql_antmaze 
