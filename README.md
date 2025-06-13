<div align="center">

## MAGREF: Masked Guidance for Any-Reference Video Generation


<a href="https://magref-video.github.io/magref.github.io/"><img src="https://img.shields.io/static/v1?label=Project&message=Page&color=blue&logo=github-pages"></a> &ensp;
<a href="https://magref-video.github.io/magref.github.io/"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%96%20Released&message=Models&color=green"></a> &ensp;
<a href="https://magref-video.github.io/magref.github.io/"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Demo&color=orange"></a> &ensp;

</div>

![teaser](./assets/teaser.png)
> **Abstract:** *Video generation has made substantial strides with the emergence of deep generative models, especially diffusion-based approaches. However, video generation based on multiple reference subjects still faces significant challenges in maintaining multi-subject consistency and ensuring high generation quality. In this paper, we propose **MAGREF**, a unified framework for any-reference video generation that introduces masked guidance to enable coherent multi-subject video synthesis conditioned on diverse reference images and a textual prompt. Specifically, we propose (1) a region-aware dynamic masking mechanism that enables a single model to flexibly handle various subject inference, including humans, objects, and backgrounds, without architectural changes, and (2) a pixel-wise channel concatenation mechanism that operates on the channel dimension to better preserve appearance features. Our model delivers state-of-the-art video generation quality, generalizing from single-subject training to complex multi-subject scenarios with coherent synthesis and precise control over individual subjects, outperforming existing open-source and commercial baselines. To facilitate evaluation, we also introduce a comprehensive multi-subject video benchmark. Extensive experiments demonstrate the effectiveness of our approach, paving the way for scalable, controllable, and high-fidelity multi-subject video synthesis.*

## 🔥 News
* `[2025.06.13]`  🔥 We will open-source the code and model weights later this month. Stay Tuned ！ 

* `[2025.05.30]`  🔥 Our arXiv paper on MAGREF is now available.  The [Project Page](https://magref-video.github.io/magref.github.io/) of MAGREF is created.


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

# 3. Install PyTorch and other dependencies using conda
# CUDA 12.1
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.4
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124


# 4. Install pip dependencies
pip install -r requirements.txt
```


### Download MAGREF Checkpoint

```bash

# if you are in china mainland, run this first: export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --repo-type model xxxx --local-dir ckpts

```


## Quick Start
- Single-GPU inference

- Multi-GPU inference using FSDP + xDiT USP
