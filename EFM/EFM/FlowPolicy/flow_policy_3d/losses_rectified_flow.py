"""Exact RectifiedFlow loss functions from ImageGeneration repository for perfect FlowPolicy training"""

import torch
import torch.optim as optim
import numpy as np
from flow_policy_3d.models import utils as mutils
from flow_policy_3d.sde_lib import RectifiedFlow


def get_rectified_flow_loss_fn(sde, train, reduce_mean=True, eps=1e-3):
  """Create a loss function for training with rectified flow - EXACT implementation from ImageGeneration.

  Args:
    sde: An `sde_lib.RectifiedFlow` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    """Compute the loss function.

    Args:
      model: A velocity model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    if sde.reflow_flag:
        z0 = batch[0]
        data = batch[1]
        batch = data.detach().clone()
    else:
        z0 = sde.get_z0(batch, train=train).to(batch.device)
    
    if sde.reflow_flag:
        if sde.reflow_t_schedule=='t0': ### distill for t = 0 (k=1)
            t = torch.zeros(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        elif sde.reflow_t_schedule=='t1': ### reverse distill for t=1 (fast embedding)
            t = torch.ones(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        elif sde.reflow_t_schedule=='uniform': ### train new rectified flow with reflow
            t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        elif type(sde.reflow_t_schedule)==int: ### k > 1 distillation
            t = torch.randint(0, sde.reflow_t_schedule, (batch.shape[0], ), device=batch.device) * (sde.T - eps) / sde.reflow_t_schedule + eps
        else:
            assert False, 'Not implemented'
    else:
        ### standard rectified flow loss
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps

    # Expand time to match batch dimensions 
    if len(batch.shape) == 4:  # Image-like data (B, C, H, W)
        t_expand = t.view(-1, 1, 1, 1).repeat(1, batch.shape[1], batch.shape[2], batch.shape[3])
    elif len(batch.shape) == 3:  # Action-like data (B, T, D) 
        t_expand = t.view(-1, 1, 1).repeat(1, batch.shape[1], batch.shape[2])
    else:
        # Handle other dimensions generically
        expand_dims = [1] * (len(batch.shape) - 1)
        t_expand = t.view(-1, *expand_dims)
        for i in range(1, len(batch.shape)):
            t_expand = t_expand.repeat(*(1,) * i + (batch.shape[i],) + (1,) * (len(batch.shape) - i - 1))
    
    perturbed_data = t_expand * batch + (1.-t_expand) * z0
    target = batch - z0 
    
    model_fn = mutils.get_model_fn(model, train=train)
    score = model_fn(perturbed_data, t*999) ### Copy from models/utils.py 

    if sde.reflow_flag:
        ### we found LPIPS loss is the best for distillation when k=1; but good to have a try
        if sde.reflow_loss=='l2':
            ### train new rectified flow with reflow or distillation with L2 loss
            losses = torch.square(score - target)
        elif sde.reflow_loss=='lpips':
            assert sde.reflow_t_schedule=='t0'
            losses = sde.lpips_model(z0 + score, batch)
        elif sde.reflow_loss=='lpips+l2':
            assert sde.reflow_t_schedule=='t0'
            lpips_losses = sde.lpips_model(z0 + score, batch).view(batch.shape[0], 1)
            l2_losses = torch.square(score - target).view(batch.shape[0], -1).mean(dim=1, keepdim=True)
            losses = lpips_losses + l2_losses
        else:
            assert False, 'Not implemented'
    else:
        losses = torch.square(score - target)
    
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn


def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.RectifiedFlow` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  if not continuous:
    if isinstance(sde, RectifiedFlow):
      loss_fn = get_rectified_flow_loss_fn(sde, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")
  else:
    raise NotImplementedError("Continuous training not implemented in this exact RectifiedFlow version")

  def step_fn(state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    if train:
      optimizer = state['optimizer']
      optimizer.zero_grad()
      loss = loss_fn(model, batch)
      loss.backward()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        loss = loss_fn(model, batch)
        ema.restore(model.parameters())

    return loss

  return step_fn