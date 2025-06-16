import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from .magref_14B import magref_14B

MAGREF_CONFIGS = {
    'magref-14B': magref_14B,
}