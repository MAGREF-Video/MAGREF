<div align="center">

## MAGREF: Masked Guidance for Any-Reference Video Generation with Subject Disentanglement

<a href="https://arxiv.org/abs/2505.23742"><img src="https://img.shields.io/badge/Arxiv-2505.20292-b31b1b.svg?logo=arXiv"></a> &ensp;
<a href="http://magref-video.github.io/"><img src="https://img.shields.io/static/v1?label=Project&message=Page&color=blue&logo=github-pages"></a> &ensp;
<a href="https://huggingface.co/MAGREF-Video/MAGREF/tree/main"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%96%20HuggingFace&message=Models&color=green"></a> &ensp;
<a href="https://modelscope.cn/models/MAGREF/magref/files"><img src="https://img.shields.io/static/v1?label=%F0%9F%9A%80%20ModelScope&message=Models&color=brightgreen"></a> &ensp;

[Yufan Deng](http://magref-video.github.io/), 
[Yuanyang Yin](https://github.com/YYY-MMW), 
[Xun Guo](https://scholar.google.com/citations?user=XtHtIDcAAAAJ&hl=en), 
[Yizhi Wang](https://actasidiot.github.io/), 
[Jacob Zhiyuan Fang](https://www.jacobzhiyuanfang.me/), <br>
[Shenghai Yuan](https://shyuanbest.github.io/), 
[Yiding Yang](https://ihollywhy.github.io/), 
[Angtian Wang](https://angtianwang.github.io/), 
[Bo Liu](http://www.svcl.ucsd.edu/people/liubo/), 
[Haibin Huang](https://brotherhuang.github.io/), 
[Chongyang Ma](http://chongyangma.com/)




> <br>Intelligent Creation Team, ByteDance<br>

</div>





![teaser](./assets/teaser.png)


## 🔥 News
* `[2025.10.10]`  🔥 Our [Research Paper](https://arxiv.org/abs/2505.23742) of MAGREF is now available.  The [Project Page](https://magref-video.github.io/) of MAGREF is created.
* `[2025.06.20]` 🙏 Thanks to **Kijai** for developing the [**ComfyUI nodes**](https://github.com/kijai/ComfyUI-WanVideoWrapper) for MAGREF and FP8-quantized Hugging Face mode! Feel free to try them out and add MAGREF to your workflow.

* `[2025.06.18]`  🔥 In progress. We are actively collecting and processing more diverse datasets and scaling up training with increased computational resources to further improve resolution, temporal consistency, and generation quality.
 Stay turned！

* `[2025.06.16]`  🔥 MAGREF is coming! The inference codes and [checkpoint](https://huggingface.co/MAGREF-Video/MAGREF/tree/main) have been released.

## 🎥 Demo



https://github.com/user-attachments/assets/ea8f7195-4ffc-4866-b210-f66bac993b7a




## 📑 Todo List
- [x] Inference codes of MAGREF-480P
- [x] Checkpoint of MAGREF-480P
- [ ] Checkpoint of MAGREF-14B Pro
- [ ] Training codes of MAGREF


## ✨ Community Works
### ComfyUI
Thanks for Kijai develop the ComfyUI nodes for MAGREF:
[https://github.com/kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)

FP8 quant Huggingface Mode: [https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan2_1-Wan-I2V-MAGREF-14B_fp8_e4m3fn.safetensors](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan2_1-Wan-I2V-MAGREF-14B_fp8_e4m3fn.safetensors)

### Guideline
Guideline by Benji: [https://www.youtube.com/watch?v=rwnh2Nnqje4](https://www.youtube.com/watch?v=rwnh2Nnqje4)


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

## 📧 Ethics Concerns
The images used in these demos are sourced from public domains or generated by models, and are intended solely to showcase the capabilities of this research. If you have any concerns, please contact us at dengyufan10@stu.pku.edu.cn, and we will promptly remove them.

## ✏️ Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:.

### BibTeX
```bibtex
@article{deng2025magref,
  title={MAGREF: Masked Guidance for Any-Reference Video Generation},
  author={Deng, Yufan and Guo, Xun and Yin, Yuanyang and Fang, Jacob Zhiyuan and Yang, Yiding and Wang, Yizhi and Yuan, Shenghai and Wang, Angtian and Liu, Bo and Huang, Haibin and others},
  journal={arXiv preprint arXiv:2505.23742},
  year={2025}
}
```
