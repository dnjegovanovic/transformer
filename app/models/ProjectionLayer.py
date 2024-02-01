import torch
import torch.nn as nn


class ProjectionLayer(nn.Module):
    """Project output from decoder to the spece of words

    Args:
        nn (_type_): _description_
    """

    def __init__(self, d_model: int, voc_size: int) -> None:
        super().__init__()
        self.proj_layer = nn.Linear(d_model, voc_size)

    def forward(self, x):
        """project x -> vocab size

        x (batch, seq_len, d_model) -> (batch, seq_len, voc_size)

        Args:
            x (_type_): _description_
        """

        # Applay softmax for numerical stability
        return torch.log_softmax(self.proj_layer(x), dim=-1)
