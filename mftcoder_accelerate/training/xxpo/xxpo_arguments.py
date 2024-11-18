"""
# @author Chaoyu Chen
# @date 2023/10/19

training arguments
"""

from dataclasses import dataclass, asdict
from typing import List, Union


@dataclass
class XXPOTrainArgs:
    # train data paths on shared FS
    data_paths: Union[str, List[str]]

    # output dir for saving adaptors in peft or full ckpts in full-parameter training
    output_dir: str

    # tensorboard dir for saving tensorboard logs
    tb_dir: str

    # pretrained_model_path, on which is the model you want to train
    pretrained_model_path: str

    # model type of pretrained_model_path, support llama|qwen|starcoder|baichuan|chatglm2
    model_type: str

    # train/valid/test split
    data_split: str = "98,2,0"

    # lora or qlora or None(for full-parameter training)
    peft_type: Union[None, str] = "qlora"

    # if qlora, 4bit will be set, else None
    quantization: Union[None, str] = "4bit"

    # lora rank, the bigger, the more trainalbe parameters
    lora_rank: int = 96

    # lora alpha
    lora_alpha: int = 32

    # lora dropout
    lora_dropout: float = 0.05

    # lora targeting modules
    target_modules: Union[None, str, List[str]] = None

    # dpo or orpo
    xxpo: str = "dpo"

    # dpo/orpo beta
    beta: float = 0.1

    rpo_alpha: Union[None, float] = None

    # mircro train batch size
    per_device_train_batch_size: int = 8

    # micro eval batch size, always same as micro train batch size
    per_device_eval_batch_size: int = 8

    # HF AutoTokenizer is supported, maybe more types
    tokenizer_type: str = "AutoTokenizer"

    # initial lr
    learning_rate: float = 5e-5

    # minimum lr
    min_lr: float = 5e-6

    # weight decay
    weight_decay: float = 0.01

    # gradient_accumulation_steps
    gradient_accumulation_steps: int = 1

    # lr_scheduler_type
    lr_scheduler_type: str = "cosine"

    # optimizer_type
    optimizer_type: str = "adamw_torch"
    # optimizer_type: str = "paged_adamw_32bit"

    # gradient_checkpointing
    gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False

    # num of warmup_steps
    warmup_steps: Union[int, float] = 0.05

    # num_train_epochs
    num_train_epochs: int = 4

    # seed for reproducing
    seed: int = 1234

    # seq_length, context length
    seq_length: int = 4096

    save_only_model: bool = True

    # path of adaptor which is resumed from, None for not resuming training
    resume_from_checkpoint: Union[None, str] = None

    # auto resume from latest ckpt if job restarted
    auto_resume: bool = True

    # num of steps for logging training loss
    logging_steps: int = 10

    # num of steps for saving ckpt
    save_steps: int = 100

    # num of steps for evaluation(eval_loss), better same as checkpointing steps
    eval_steps: int = 100

    # max train steps, if None, depends on num_train_epochs
    max_steps: int = -1

    # if checkpointing every epoch, maybe True in sst
    epoch_checkpointing: bool = False

    # shuffle before train/valid split
    shuffle_before_split: bool = True

    # if early stop when eval loss is not converging in the past early_stopping_stall_num evaluation point
    early_stopping: bool = True
    early_stopping_stall_num: int = 5

    # limit num for saving ckpts, None for no limits. Used for full-parameter training to avoid exceeding disk quota.
    saving_limit: Union[None, int] = None

    # ATTENTION_CLASSES = { "eager": Normal Attention, "flash_attention_2": FlashAttention2}
    attn_implementation: str = "flash_attention_2"

    # tokenizer chat template, if None, will use MFTCoder template
    chat_template: Union[None, str] = None

    distributed_type: Union[None, str] = None

    init_timeout_seconds: Union[None, int] = 3600

    make_vocab_size_divisible_by: int = 32
    model_parallel_size: int = 1
    use_slow_tokenizer: bool = False
    world_size: int = 8

    # max prompt string length and whole str length
    max_prompt_length: Union[None, int] = 2048
    max_length: Union[None, int] = 4096

    # num of process processing dataset
    dataset_num_proc: int = 1

    # model_dtype[float16, bfloat16, float] for loading
    dtype: str = "bfloat16"

    # instrumentation
    disable_tqdm: bool = False
    sanity_check: bool = False

    # debug argument for distributed training
    # "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
    # "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
    ignore_bias_buffers: bool = True

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}
