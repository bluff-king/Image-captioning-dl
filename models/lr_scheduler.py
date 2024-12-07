import torch
import math


def linear_warmup_decay(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return (step + 1) / (warmup_steps + 1)
    else:
        return max(1e-7, (total_steps - step) / (total_steps - warmup_steps))


def warmup_cosine_with_restarts(step, warmup_steps, total_steps, num_cycles=1):
    if step < warmup_steps:
        return (step + 1) / (warmup_steps + 1)
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cycle_progress = progress * num_cycles % 1
        return max(1e-7, 0.5 * (1 + math.cos(math.pi * cycle_progress)))



def get_scheduler(
    optimizer, total_steps, warmup_steps, num_cycles=None, types='warmup_cosine_with_restarts'
):
    if types == 'warmup_cosine_with_restarts':
        assert num_cycles != None, 'must specify num_cycles when types="warmup_cosine_with_restarts"'
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: warmup_cosine_with_restarts(
                step, warmup_steps, total_steps, num_cycles=num_cycles)
        )
    elif types == 'linear_warmup_decay':
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: linear_warmup_decay(step, warmup_steps, total_steps)
        )
    else:
        raise Exception('not implemented')
