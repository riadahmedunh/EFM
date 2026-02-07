from typing import Union, Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LinearLR

def get_scheduler(
    name: Union[str],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs
):
    """
    Simplified scheduler function to avoid diffusers dependency issues.
    
    Args:
        name (`str`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do.
        num_training_steps (`int``, *optional*):
            The number of training steps to do.
    """
    if name == "constant":
        # Return a dummy scheduler that doesn't change LR
        return StepLR(optimizer, step_size=10000000, gamma=1.0)
    elif name == "cosine":
        if num_training_steps is None:
            raise ValueError("cosine scheduler requires num_training_steps")
        return CosineAnnealingLR(optimizer, T_max=num_training_steps)
    elif name == "linear":
        if num_training_steps is None:
            raise ValueError("linear scheduler requires num_training_steps")
        return LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps)
    else:
        # Default to constant
        return StepLR(optimizer, step_size=10000000, gamma=1.0)
