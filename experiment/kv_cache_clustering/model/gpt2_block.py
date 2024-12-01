from torch import nn

class GPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super.__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size