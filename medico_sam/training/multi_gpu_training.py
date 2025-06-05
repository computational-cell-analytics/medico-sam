from functools import partial
from typing import Dict, Any, Callable

import torch
import torch.utils.data

from torch_em.multi_gpu_training import setup, cleanup, _create_data_loader, DDP

from micro_sam.instance_segmentation import get_unetr


def _train_impl(
    rank: int,
    world_size: int,
    model_callable: Callable[[Any], torch.nn.Module],
    model_kwargs: Dict[str, Any],
    train_dataset_callable: Callable[[Any], torch.utils.data.Dataset],
    train_dataset_kwargs: Dict[str, Any],
    val_dataset_callable: Callable[[Any], torch.utils.data.Dataset],
    val_dataset_kwargs: Dict[str, Any],
    loader_kwargs: Dict[str, Any],
    iterations: int,
    optimizer_callable: Callable[[Any], torch.optim.Optimizer],
    optimizer_kwargs: Dict[str, Any],
    lr_scheduler_callable: Callable[[Any], torch.optim.lr_scheduler._LRScheduler],
    lr_scheduler_kwargs: Dict[str, Any],
    trainer_callable: Callable,
    find_unused_parameters: bool = True,
    **kwargs
):
    assert "device" not in kwargs
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)

    model, state = model_callable(return_state=True, **model_kwargs)
    unetr = get_unetr(image_encoder=model.sam.image_encoder, decoder_state=state.get("decoder_state", None))

    model.to(rank)
    unetr.to(rank)

    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=find_unused_parameters)
    ddp_unetr = DDP(unetr, device_ids=[rank], find_unused_parameters=find_unused_parameters)

    model_params = [params for params in model.parameters()]
    decoder_params = [params for param_name, params in unetr.named_parameters() if not param_name.startswith("encoder")]

    optimizer = optimizer_callable(model_params + decoder_params, **optimizer_kwargs)
    kwargs["optimizer"] = optimizer

    lr_scheduler = lr_scheduler_callable(optimizer, **lr_scheduler_kwargs)
    kwargs["lr_scheduler"] = lr_scheduler

    train_loader = _create_data_loader(train_dataset_callable, train_dataset_kwargs, loader_kwargs, world_size, rank)
    val_loader = _create_data_loader(val_dataset_callable, val_dataset_kwargs, loader_kwargs, world_size, rank)

    trainer = trainer_callable(
        model=ddp_model,
        unetr=ddp_unetr,
        train_loader=train_loader,
        val_loader=val_loader,
        device=rank,
        rank=rank,
        **kwargs
    )
    trainer.fit(iterations=iterations)

    cleanup()


def train_multi_gpu(
    model_callable: Callable[[Any], torch.nn.Module],
    model_kwargs: Dict[str, Any],
    train_dataset_callable: Callable[[Any], torch.utils.data.Dataset],
    train_dataset_kwargs: Dict[str, Any],
    val_dataset_callable: Callable[[Any], torch.utils.data.Dataset],
    val_dataset_kwargs: Dict[str, Any],
    loader_kwargs: Dict[str, Any],
    iterations: int,
    optimizer_callable: Callable[[Any], torch.optim.Optimizer],
    optimizer_kwargs: Dict[str, Any],
    lr_scheduler_callable: Callable[[Any], torch.optim.lr_scheduler._LRScheduler],
    lr_scheduler_kwargs: Dict[str, Any],
    trainer_callable: Callable,
    find_unused_parameters: bool = True,
    **kwargs
) -> None:
    """Run data parallel training on multiple local GPUs via torch.distributed.

    This function will run training on all available local GPUs in parallel.
    To use it, the function / classes and keywords for the model and data loaders must be given.
    Optionaly, functions / classes and keywords for the optimizer, learning rate scheduler and trainer class
    may be given, so that they can be instantiated for each training child process.

    Args:
        model_callable: Function or class to create the model.
        model_kwargs: Keyword arguments for `model_callable`.
        train_dataset_callable: Function or class to create the training dataset.
        train_dataset_kwargs: Keyword arguments for `train_dataset_callable`.
        val_dataset_callable: Function or class to create the validation dataset.
        val_dataset_kwargs: Keyword arguments for `val_dataset_callable`.
        loader_kwargs: Keyword arguments for the torch data loader.
        iterations: Number of iterations to train for.
        optimizer_callable: Function or class to create the optimizer.
        optimizer_kwargs: Keyword arguments for `optimizer_callable`.
        lr_scheduler_callable: Function or class to create the learning rate scheduler.
        lr_scheduler_kwargs: Keyword arguments for `lr_scheduler_callable`.
        trainer_callable: Function or class to create the trainer.
        find_unused_parameters: Whether to find unused parameters of the model to exclude from the optimization.
        kwargs: Keyword arguments for `trainer_callable`.
    """
    world_size = torch.cuda.device_count()
    train = partial(
        _train_impl,
        model_callable=model_callable,
        model_kwargs=model_kwargs,
        train_dataset_callable=train_dataset_callable,
        train_dataset_kwargs=train_dataset_kwargs,
        val_dataset_callable=val_dataset_callable,
        val_dataset_kwargs=val_dataset_kwargs,
        loader_kwargs=loader_kwargs,
        iterations=iterations,
        optimizer_callable=optimizer_callable,
        optimizer_kwargs=optimizer_kwargs,
        lr_scheduler_callable=lr_scheduler_callable,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        trainer_callable=trainer_callable,
        find_unused_parameters=find_unused_parameters,
        **kwargs
    )
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
