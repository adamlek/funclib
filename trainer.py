from typing import (
    Callable, 
    Any, 
    Tuple, 
    Iterable, 
    )

import torch
import torch.nn as nn

def train_step(model, batch, labels, optimizer, loss_f):
    logits = model(**batch)
    loss = loss_f(logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

def torch_trainer(
    model: Any,
    dataset: list[Any],
    dataset_sampler: Callable, 
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_f: Callable,
    train_step_f: Callable,
    ) -> bool:

    return True