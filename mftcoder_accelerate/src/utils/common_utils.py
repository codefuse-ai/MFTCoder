import os
import math
import torch

TASK2ID = {}
ID2TASK = {}

def print_rank_0(*message):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*message, flush=True)
    else:
        print(*message, flush=True)


def wait_for_everyone():
    torch.distributed.barrier()


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


def generate_task_id(data_paths):
    data_prefixes = list(data_paths[1:-1].split(','))
    print("data paths: ")
    print(data_prefixes)

    for i, prefix in enumerate(data_prefixes):
        task_name = prefix.split('/')[-1]
        TASK2ID[task_name] = i
        ID2TASK[i] = task_name
