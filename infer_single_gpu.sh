#!/bin/bash

save_dir=./samples/${now}
mkdir -p "${save_dir}"


## single_id driven video generation
python generate.py \
    --ckpt_dir ./ckpts/magref \
    --save_dir "${save_dir}" \
    --prompt_path assets/single_id.txt \



## multi_id driven video generation
python generate.py \
    --ckpt_dir ./ckpts/magref \
    --save_dir "${save_dir}" \
    --prompt_path assets/multi_id.txt \



## id_obj_env driven video generation
python generate.py \
    --ckpt_dir ./ckpts/magref \
    --save_dir "${save_dir}" \
    --prompt_path assets/id_obj_env.txt \
