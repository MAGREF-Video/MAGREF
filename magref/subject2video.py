import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

from PIL import Image, ImageOps

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class MagRefModel:
    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
    ):
        r"""
        Initializes the subject-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            if not init_on_cpu:
                self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def process_refimg(self, pil_images, target_height=480, target_width=832):
        """
        处理参考图像，支持多种 scale 布局方式。
        根据输入图像数量自动选择合适的 scale。

        Args:
            pil_images: 输入的 PIL 图像列表
            target_height: 目标高度
            target_width: 目标宽度

        Returns:
            - PIL.Image: 拼接后的图像
            - torch.Tensor: 下采样后的 latent mask
            - torch.Tensor: latent blocks 坐标
            - list: mllm_refimg 图像列表
        """
        # 根据图像数量自动选择 scale
        num_images = len(pil_images)
        available_scales = [1, 2, 3, 4, 6, 8]
        scale = min([s for s in available_scales if s >= num_images], default=8)

        if self.rank == 0:
            logging.info(f"自动选择 scale={scale} (输入图像数量: {num_images})")
        # 1. 构建白底大图和对应 mask
        canvas = Image.new("RGB", (target_width, target_height), (255, 255, 255))
        mask_np = np.zeros((target_height, target_width), dtype=np.uint8)
        mllm_refimg = []

        # Determine the number of blocks based on the scale
        if scale == 1:
            block_layout = [(0, 0, target_width, target_height)]
        elif scale == 2:
            block_width = target_width // 2
            block_layout = [(0, 0, block_width, target_height),
                          (block_width, 0, target_width, target_height)]
        elif scale == 3:
            block_width = target_width // 3
            block_layout = [(0, 0, block_width, target_height),
                          (block_width, 0, block_width * 2, target_height),
                          (block_width * 2, 0, target_width, target_height)]
        elif scale == 4:
            block_width = target_width // 2
            block_height = target_height // 2
            block_layout = [(0, 0, block_width, block_height),
                          (block_width, 0, target_width, block_height),
                          (0, block_height, block_width, target_height),
                          (block_width, block_height, target_width, target_height)]
        elif scale == 6:
            block_width = target_width // 3
            block_height = target_height // 2
            block_layout = [(0, 0, block_width, block_height),
                          (block_width, 0, block_width * 2, block_height),
                          (block_width * 2, 0, target_width, block_height),
                          (0, block_height, block_width, target_height),
                          (block_width, block_height, block_width * 2, target_height),
                          (block_width * 2, block_height, target_width, target_height)]
        elif scale == 8:
            block_width = target_width // 4
            block_height = target_height // 2
            block_layout = [(0, 0, block_width, block_height),
                          (block_width, 0, block_width * 2, block_height),
                          (block_width * 2, 0, block_width * 3, block_height),
                          (block_width * 3, 0, target_width, block_height),
                          (0, block_height, block_width, target_height),
                          (block_width, block_height, block_width * 2, target_height),
                          (block_width * 2, block_height, block_width * 3, target_height),
                          (block_width * 3, block_height, target_width, target_height)]

        # Randomly distribute images into the blocks
        num_blocks = len(block_layout)
        indices = list(range(len(pil_images)))
        random.shuffle(indices)
        random.shuffle(block_layout)

        # Ensure we don't use more images than blocks
        indices = indices[:num_blocks]

        for img_i, (x_start, y_start, x_end, y_end) in zip(indices, block_layout):
            pil_img = pil_images[img_i]
            block_w = x_end - x_start
            block_h = y_end - y_start

            # Resize and pad image to fit the block
            only_resize_img, resized_img, resized_mask = self.resize_and_pad(
                pil_img, target_height=block_h, target_width=block_w)
            img_pil = Image.fromarray(resized_img.numpy().transpose(1, 2, 0))
            mask_np[y_start:y_end, x_start:x_end] = resized_mask.numpy()
            canvas.paste(img_pil, (x_start, y_start))

            img_tensor = torch.from_numpy(np.array(only_resize_img)).permute(2, 0, 1).contiguous()
            mllm_refimg.append(img_tensor)

        # 2. 最终返回拼接好的图像（转换为 tensor）和对应 mask
        np_img = np.asarray(canvas, dtype=np.uint8).transpose(2, 0, 1)
        out_tensor = torch.from_numpy(np_img)
        mask_tensor = torch.from_numpy(mask_np)

        # 3. 下采样 mask（stride=8）
        def downsample_mask(mask_np, factor=8):
            h, w = mask_np.shape
            new_h, new_w = h // factor, w // factor
            downsampled = np.zeros((new_h, new_w), dtype=np.uint8)
            for i in range(new_h):
                for j in range(new_w):
                    patch = mask_np[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor]
                    if patch.any():
                        downsampled[i, j] = 1
            return torch.from_numpy(downsampled)

        def get_latent_regions_safe(blocks, factor=8):
            latent_blocks = []
            for x_start, y_start, x_end, y_end in blocks:
                latent_x_end = x_end // factor
                latent_x_start = math.ceil(x_start / factor) if x_start > 0 else 0
                latent_y_end = y_end // factor
                latent_y_start = math.ceil(y_start / factor) if y_start > 0 else 0

                if latent_x_start < latent_x_end and latent_y_start < latent_y_end:
                    latent_blocks.append((latent_x_start, latent_x_end, latent_y_start, latent_y_end))

            if not latent_blocks:
                return torch.empty((0, 4), dtype=torch.long)
            return torch.tensor(latent_blocks, dtype=torch.long)

        latent_mask_tensor = downsample_mask(mask_np)
        latent_blocks = get_latent_regions_safe(block_layout)

        out_pil = Image.fromarray(out_tensor.permute(1, 2, 0).numpy())

        # Save debug images if enabled
        if hasattr(self.config, 'save_debug_images') and self.config.save_debug_images:
            debug_dir = getattr(self.config, 'debug_dir', 'debug_output')
            os.makedirs(debug_dir, exist_ok=True)

            # Generate unique filename based on timestamp
            import time
            timestamp = int(time.time() * 1000)

            # Save composed image
            composed_path = os.path.join(debug_dir, f'composed_image_{timestamp}.jpg')
            out_pil.save(composed_path, format='JPEG')

            # Save mask as grayscale image
            mask_image = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')
            mask_path = os.path.join(debug_dir, f'mask_image_{timestamp}.jpg')
            mask_image.save(mask_path, format='JPEG')

            if self.rank == 0:
                logging.info(f"Saved debug images: {composed_path}, {mask_path}")

        return out_pil, latent_mask_tensor, latent_blocks, mllm_refimg
    

    def resize_and_pad(self, pil_img: Image.Image, target_height: int, target_width: int):
        """
        等比缩放 + 白色居中填充，并返回原始缩放图像、填充后的图像和有效区域 mask。

        Returns:
            - PIL.Image: 仅缩放的图像（未填充）
            - torch.Tensor(uint8): 填充后的图像张量 (3, H, W)
            - torch.Tensor(uint8): mask张量 (H, W)，1表示有效区域，0表示填充区域
        """
        if pil_img.mode == 'RGBA':
            background = Image.new("RGB", pil_img.size, (255, 255, 255))
            background.paste(pil_img, mask=pil_img.split()[3])
            img = background
        else:
            img = pil_img.convert("RGB")

        # 原始尺寸缩放
        img_ratio = img.width / img.height
        target_ratio = target_width / target_height

        if img_ratio > target_ratio:
            new_w = target_width
            new_h = int(new_w / img_ratio)
        else:
            new_h = target_height
            new_w = int(new_h * img_ratio)

        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        only_resize_img = img

        # padding 参数
        pad_w = target_width - new_w
        pad_h = target_height - new_h
        padding = (
            pad_w // 2,
            pad_h // 2,
            pad_w - pad_w // 2,
            pad_h - pad_h // 2,
        )

        # 构造图像和 mask 同时填充
        img = ImageOps.expand(img, padding, fill=(255, 255, 255))
        mask = Image.new("L", (new_w, new_h), color=1)
        mask = ImageOps.expand(mask, padding, fill=0)

        # 转成张量
        np_img = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)
        np_mask = np.asarray(mask, dtype=np.uint8)

        return only_resize_img, torch.from_numpy(np_img), torch.from_numpy(np_mask)

    def generate(self,
                 input_prompt,
                 img,
                 max_area=480 * 832,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from input reference images and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image or list):
                Input image or list of images. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """

        # 统一使用新的实现，自动根据图像数量选择 scale
        if not isinstance(img, list):
            img = [img]
        img, ref_latent_mask, latent_blocks, mllm_refimg = self.process_refimg(
            img, target_height=480, target_width=832)
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]

        assert h * w <= max_area, f"h*w={h*w} must be less than max_area={max_area}"
        assert h % self.vae_stride[1] == 0, f"h={h} must be divisible by stride={self.vae_stride[1]}"
        assert w % self.vae_stride[2] == 0, f"w={w} must be divisible by stride={self.vae_stride[2]}"
        lat_h = h // self.vae_stride[1]
        lat_w = w // self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size


        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16, (F - 1) // 4 + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)
        
        msk = torch.zeros(1, 81, lat_h, lat_w, device=self.device)
        msk[:, :1] = ref_latent_mask
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        self.clip.model.to(self.device)
        clip_context = self.clip.visual([img[:, None, :, :]])
        if offload_model:
            self.clip.model.cpu()

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1), torch.zeros(3, F - 1, h, w)], dim=1).to(self.device)])[0]
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise

            arg_c = {
                'context': [context[0]],
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
            }

            arg_null = {
                'context': context_null,
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
            }

            if offload_model:
                torch.cuda.empty_cache()

            self.model.to(self.device)
            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                latent = latent.to(
                    torch.device('cpu') if offload_model else self.device)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                x0 = [latent.to(self.device)]
                del latent_model_input, timestep

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latent
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None