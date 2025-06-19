<div align="center">

## MAGREF: Masked Guidance for Any-Reference Video Generation


<a href="https://magref-video.github.io/magref.github.io/"><img src="https://img.shields.io/static/v1?label=Project&message=Page&color=blue&logo=github-pages"></a> &ensp;
<a href="https://huggingface.co/MAGREF-Video/MAGREF/tree/main"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%96%20HuggingFace&message=Models&color=green"></a> &ensp;
<a href="https://modelscope.cn/models/MAGREF/magref/files"><img src="https://img.shields.io/static/v1?label=%F0%9F%9A%80%20ModelScope&message=Models&color=brightgreen"></a> &ensp;
<a href="https://magref-video.github.io/magref.github.io/"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Demo&color=orange"></a> &ensp;

</div>

![teaser](./assets/teaser.png)


## 🔥 News
* `[2025.06.18]`  🔥 In progress. We are actively collecting and processing more diverse datasets and scaling up training with increased computational resources to further improve resolution, temporal consistency, and generation quality.
 Stay turned！

* `[2025.06.16]`  🔥 MAGREF is coming! The inference codes and [checkpoint](https://huggingface.co/MAGREF-Video/MAGREF/tree/main) have been released.

* `[2025.05.30]`  🔥 Our arXiv paper on MAGREF is now available.  The [Project Page](https://magref-video.github.io/magref.github.io/) of MAGREF is created.

## 📑 Todo List
- [x] Inference codes of MAGREF-480P
- [x] Checkpoint of MAGREF-480P
- [ ] Checkpoint of MAGREF-14B Pro
- [ ] Training codes of MAGREF

## ⚙️ Requirements and Installation
We recommend the requirements as follows.

### Environment

```bash
# 0. Clone the repo
git clone https://github.com/MAGREF-Video/MAGREF.git
cd MAGREF

# 1. Create conda environment
conda create -n magref python=3.11.2
conda activate magref

# 2. Install PyTorch and other dependencies
# CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.4
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 3. Install pip dependencies
pip install -r requirements.txt

# 4. (Optional) Install xfuser for multiple GPUs inference
pip install "xfuser>=0.4.1"
```


### Download MAGREF Checkpoint

```bash

# if you are in china mainland, run this first: export HF_ENDPOINT=https://hf-mirror.com
# pip install -U "huggingface_hub[cli]"
huggingface-cli download MAGREF-Video/MAGREF --local-dir ./ckpts/magref

```


## 🤗 Quick Start
- Single-GPU inference
<br>Tested on a single NVIDIA H100 GPU.
  The inference consumes around **70 GB** of VRAM, so an 80 GB GPU is recommended.
```bash
# way 1
bash infer_single_gpu.sh

# way 2
python generate.py \
    --ckpt_dir ./ckpts/magref \
    --save_dir ./samples \
    --prompt_path ./assets/single_id.txt \
```

- Multi-GPU inference
```bash
# way 1
bash infer_multi_gpu.sh

# way 2
torchrun --nproc_per_node=8 generate.py \
    --dit_fsdp --t5_fsdp --ulysses_size 8 \
    --ckpt_dir ./ckpts/magref \
    --save_dir ./samples \
    --prompt_path ./assets/multi_id.txt \
```
> 💡Note: 
> * To achieve the best generation results, we recommend that you describe the visual content of the reference image as accurately as possible when writing text prompt.
> * When the generated video is unsatisfactory, the most straightforward solution is to try changing the `--base_seed` and modifying the description in the prompt.


## 👍 Acknowledgement

* This project wouldn't be possible without the following open-sourced repositories: [Wan2.1](https://github.com/Wan-Video/Wan2.1), [VACE](https://github.com/ali-vilab/VACE), [Phantom](https://github.com/Phantom-video/Phantom), [SkyReels-A2](https://github.com/SkyworkAI/SkyReels-A2), [HunyuanCustom](https://github.com/Tencent-Hunyuan/HunyuanCustom), [ConsisID](https://github.com/PKU-YuanGroup/ConsisID), [Concat-ID](https://github.com/ML-GSAI/Concat-ID)


