import torch.nn as nn

def num_params(model):
    # utility function
    return sum([p.numel() for p in model.parameters()])

def init_all(model):
    # probably better ways to do this
    for _, child in model.named_children():
        if not hasattr(child, 'momentum') and hasattr(child, 'shape'):
            if hasattr(child, 'weight') and child.weight is not None:
                nn.init.xavier_uniform_(child.weight)
            if hasattr(child, 'bias') and child.bias is not None:
                nn.init.constant_(child.bias, 1.0)

    return model

