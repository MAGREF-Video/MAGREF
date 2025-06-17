#!/bin/bash

if [[ ! -d "logs" ]]; then
  mkdir logs
fi


job_name=${1-infer_magref}
echo 'start job:' ${job_name}
now=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=logs/${job_name}_${now}.log
echo 'log file: ' ${LOG_FILE}


## single_id driven video generation
save_dir=./samples/${now}
mkdir -p "${save_dir}"
torchrun --nproc_per_node=8 generate.py \
    --dit_fsdp --t5_fsdp --ulysses_size 8 \
    --ckpt_dir ./ckpts/magref \
    --save_dir "${save_dir}" \
    --prompt_path assets/single_id.txt \
    2>&1 | tee ${LOG_FILE}


# multi_id driven video generation
save_dir=./samples/${now}
mkdir -p "${save_dir}"
torchrun --nproc_per_node=8 generate.py \
    --dit_fsdp --t5_fsdp --ulysses_size 8 \
    --ckpt_dir ./ckpts/magref \
    --save_dir "${save_dir}" \
    --prompt_path assets/multi_id.txt \
    2>&1 | tee ${LOG_FILE}


# id_obj_env driven video generation
save_dir=./samples/${now}
mkdir -p "${save_dir}"
torchrun --nproc_per_node=8 generate.py \
    --dit_fsdp --t5_fsdp --ulysses_size 8 \
    --ckpt_dir ./ckpts/magref \
    --save_dir "${save_dir}" \
    --prompt_path assets/id_obj_env.txt \
    2>&1 | tee ${LOG_FILE}
