import torch
import numpy as np


def subsequent_mask(size):
    """Visualize attn mask
        plt.imshow(sub_mask)
    """
    attn_shape = (1, size, size)
    sub_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(sub_mask) == 0
