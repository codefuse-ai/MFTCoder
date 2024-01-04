import os
import math
import torch
import atorch
import numpy as np
from collections.abc import Mapping  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    # BackwardPrefetch,
    FullStateDictConfig,
    StateDictType,
)
from transformers import get_scheduler
from utils.learning_rates import AnnealingLR
TASK2ID = {}
ID2TASK = {}


def get_rank():
    return atorch.rank()


def get_local_rank():
    return atorch.local_rank()


def is_main_process():
    return atorch.rank() == 0


def is_local_main_process():
    return atorch.local_rank() == 0


def print_rank_0(*message):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*message, flush=True)
    else:
        print(*message, flush=True)


def get_world_size():
    return atorch.world_size()


def wait_for_everyone():
    torch.distributed.barrier()


def atorch_init_distributed(backend="nccl"):
    atorch.init_distributed(backend, set_cuda_device_using_local_rank=True)
    # atorch.init_distributed(backend)


def atorch_reset_distributed():
    atorch.reset_distributed()


def _goes_first(is_main):
    if is_main is False:
        wait_for_everyone()
    yield
    if is_main is True:
        wait_for_everyone()


def get_model_params_num(model):
    """
    Get params number of the model
    Args:
        model: model(required)
    Returns:
        the number of parameters of model
    """
    num = 0
    for _, param in model.named_parameters():
        num += param.nelement()
    return num


@contextmanager
def main_process_first():
    yield from _goes_first(is_main_process())


def unwrap_model(model):
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


def honor_type(obj, generator):
    """
    Cast a generator to the same type as obj (list, tuple or namedtuple)
    """
    try:
        return type(obj)(generator)
    except TypeError:
        # Some objects may not be able to instantiate from a generator directly
        return type(obj)(*list(generator))


def recursively_apply(
    func,
    data,
    *args,
    test_type=lambda t: isinstance(t, torch.Tensor),
    error_on_other_type=False,
    **kwargs,
):
    if isinstance(data, (tuple, list)):
        return honor_type(
            data,
            (
                recursively_apply(
                    func,
                    o,
                    *args,
                    test_type=test_type,
                    error_on_other_type=error_on_other_type,
                    **kwargs,
                )
                for o in data
            ),
        )
    elif isinstance(data, Mapping):
        return type(data)(
            {
                k: recursively_apply(
                    func,
                    v,
                    *args,
                    test_type=test_type,
                    error_on_other_type=error_on_other_type,
                    **kwargs,
                )
                for k, v in data.items()
            }
        )
    elif test_type(data):
        return func(data, *args, **kwargs)
    elif error_on_other_type:
        raise TypeError(
            f"Can't apply {func.__name__} on object of type {type(data)}, only of nested list/tuple/dicts of objects "
            f"that satisfy {test_type.__name__}."
        )
    return data


def gather(tensor):
    def _gpu_gather_one(tensor):
        if tensor.ndim == 0:
            tensor = tensor.clone()[None]
        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)
        return torch.cat(output_tensors, dim=0)

    return recursively_apply(_gpu_gather_one, tensor, error_on_other_type=True)


def save_ckpt(model, optimizer, lr_scheduler, epoch, steps, save_path, logger):
    if isinstance(model, FSDP):
        print('Saving a FSDP model')
        optim_state_dict = FSDP.full_optim_state_dict(model, optimizer)
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            model_state_dict = model.state_dict()
        lrs_state_dict = lr_scheduler.state_dict()
    else:
        print('Saving a normal model')
        model_state_dict = model.state_dict()
        optim_state_dict = optimizer.state_dict()
        lrs_state_dict = lr_scheduler.state_dict()
    # rank0 save
    if is_main_process():
        torch.save(
            {
                "epoch": epoch + 1,
                "step": steps,
                "state_dict": model_state_dict,
                "optimizer": optim_state_dict,
                "lrs_state_dict": lrs_state_dict,
            },
            save_path,
        )
        logger.info(f"Saved checkpoint {save_path} (epoch {epoch + 1} @ {steps} steps)")
    wait_for_everyone()
    # torch.distributed.barrier()  # other rank waiting


def scheduler_and_resume(args, train_dataloader, model, optimizer, checkpoint):
    # Scheduler and math around the number of training steps.
    overrode_max_steps = False
    args.num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_steps == -1:
        args.max_steps = args.num_train_epochs * args.num_update_steps_per_epoch
        overrode_max_steps = True

    lr_scheduler = AnnealingLR(
        optimizer,
        start_lr=args.learning_rate,
        warmup_iter=args.num_warmup_steps,
        total_iters=args.max_steps * args.gradient_accumulation_steps,
        decay_style=args.lr_scheduler_type,
        last_iter=0,
        min_lr=args.min_lr,
        use_checkpoint_lr_scheduler=True,
    )

    if args.resume_from_checkpoint is not None:
        if os.path.isfile(args.resume_from_checkpoint):
            starting_epoch = checkpoint["epoch"] - 1
            steps = checkpoint["step"]
            args.resume_step = steps
            # Restore the optim state
            if optimizer is not None:
                if isinstance(model, FSDP):
                    print('Loading optimizer for a FSDP model')
                    full_osd = checkpoint["optimizer"]
                    sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)
                    optimizer.load_state_dict(sharded_osd)
                else:
                    print('Loading optimizer for a normal model')
                    optimizer.load_state_dict(checkpoint["optimizer"])
                logging.info("Optimizer state is restored from the checkpoint")
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(checkpoint["lrs_state_dict"])
            logging.info(f"Loaded checkpoint '{args.resume_from_checkpoint}' (epoch {checkpoint['epoch']} @ {steps} steps)")
        else:
            logger.info(f"No optimizer and lr scheduler checkpoint found at '{args.resume_from_checkpoint}'")

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    args.num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_steps:
        args.max_steps = args.num_train_epochs * args.num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_steps / args.num_update_steps_per_epoch)
    
    return args, lr_scheduler, optimizer


def get_computation_speed(batch_size_per_device, seq_len, step_time):

    return batch_size_per_device * seq_len / (step_time + 1e-12)


def human_readable_flops(num):
    for unit in [
        "",
        "KFLOPS",
        "MFLOPS",
        "GFLOPS",
        "TFLOPS",
        "PFLOPS",
        "EFLOPS",
        "ZFLOPS",
    ]:
        if abs(num) < 1000.0:
            return "%3.1f%s" % (num, unit)
        num /= 1000.0
    return "%.1f%s" % (num, "Yi")


def get_tflops_new(args, batch_size, seq_len, step_time):
    sl = seq_len
    L = args.num_hidden_layers
    h = args.hidden_size
    V = args.vocab_size
    flops = (96 * batch_size * sl * L * h * h * (1 + sl / (6 * h) + V / (16 * L * h)) / step_time)
    return human_readable_flops(flops)


def get_tflops_megatron(total_model_param, hidden_size, num_hidden_layers, 
                        batch_size_per_device, seq_len, step_time):

    ff = total_model_param * 6
    attn = seq_len * hidden_size * num_hidden_layers * 60
    flops = (
        batch_size_per_device
        * seq_len
        * (ff + attn)
        / step_time
    )
    return human_readable_flops(flops)


def is_old_version(path):
    new_vocab_files = ['merge.model']
    new_vocab_file_exists = []
    for filename in new_vocab_files:
        if not os.path.exists(os.path.join(path, filename)):
            new_vocab_file_exists.append(False)
        else:
            new_vocab_file_exists.append(True)
    if all(new_vocab_file_exists):
        return False
    if any(new_vocab_file_exists):
        return 'new_version_file_absent'
    else:
        return True


def generate_task_id(data_paths, train_mode):
    data_prefixes = list(data_paths[1:-1].split(','))
    print("data paths: ")
    print(data_prefixes)

    for i, prefix in enumerate(data_prefixes):
        if train_mode == 'sft':
            task_name = prefix.split('/')[-1]
        else:
            task_name = prefix.split('/')[-2]
        TASK2ID[task_name] = i
        ID2TASK[i] = task_name


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

