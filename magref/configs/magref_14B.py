import torch
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ MagRef 14B ------------------------#

magref_14B = EasyDict(__name__='Config: MagRef 14B')
magref_14B.update(wan_shared_cfg)
magref_14B.sample_neg_prompt = "镜头晃动，" + magref_14B.sample_neg_prompt

magref_14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
magref_14B.t5_tokenizer = 'google/umt5-xxl'

# clip
magref_14B.clip_model = 'clip_xlm_roberta_vit_h_14'
magref_14B.clip_dtype = torch.float16
magref_14B.clip_checkpoint = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
magref_14B.clip_tokenizer = 'xlm-roberta-large'

# vae
magref_14B.vae_checkpoint = 'Wan2.1_VAE.pth'
magref_14B.vae_stride = (4, 8, 8)

# transformer
magref_14B.patch_size = (1, 2, 2)
magref_14B.dim = 5120
magref_14B.ffn_dim = 13824
magref_14B.freq_dim = 256
magref_14B.num_heads = 40
magref_14B.num_layers = 40
magref_14B.window_size = (-1, -1)
magref_14B.qk_norm = True
magref_14B.cross_attn_norm = True
magref_14B.eps = 1e-6
