"""
# @author Chaoyu Chen
# @date 2023/10/19

accelerate + deepspeed zero stage2 + Data Parallelism
MFT Training
"""
from dataclasses import dataclass, asdict
from typing import List, Union


@dataclass
class TrainArgs:
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

    # load from raw jsonl file or tokenized binary file
    load_raw_dataset: bool = True

    # weights of loss calculation for each task, None means equal weights
    task_weights: Union[None, str] = None

    # weights of data sampling, leave it None
    data_weights: Union[None, str] = None

    # hf loading model low_cpu_mem_usage
    low_cpu_mem_usage: bool = True

    # train/valid/test split
    data_split: str = "98,2,0"

    # padding or pack or concat
    padding_mode: str = "padding"

    # sft or sst
    tokenize_mode: str = "sft"

    # case3 or case4
    weighted_loss_mode: str = "case3"

    # lora or qlora or None(for full-parameter training)
    peft_type: str = "qlora"

    # if qlora, 4bit or 8bit, 4bit is suggested
    quantization: str = "4bit"

    # lora rank, the bigger, the more trainalbe parameters
    lora_rank: int = 96

    # lora alpha
    lora_alpha: int = 32

    # lora dropout
    lora_dropout: float = 0.05

    # lora targeting modules
    target_modules: Union[None, List[str]] = None

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
    weight_decay: float = 0.1

    # gradient_accumulation_steps
    gradient_accumulation_steps: int = 1

    # lr_scheduler_type
    lr_scheduler_type: str = "cosine"

    # num_warmup_steps
    num_warmup_steps: int = 300

    # num_train_epochs
    num_train_epochs: int = 4

    # seed for reproducing
    seed: int = 1234

    # seq_length, context length
    seq_length: int = 4096

    # path of adaptor which is resumed from, None for not resuming training
    resume_from_checkpoint: Union[None, str] = None

    # num of steps for logging training loss
    log_interval: int = 10

    # num of steps for saving ckpt
    checkpointing_steps: int = 100

    # num of steps for evaluation(eval_loss), better same as checkpointing steps
    evaluation_steps: int = 100

    # max train steps, if None, depends on num_train_epochs
    max_train_steps: Union[None, int] = None

    # if checkpointing every epoch, maybe True in sst
    epoch_checkpointing: bool = False

    # shuffle before train/valid split
    shuffle_before_split: bool = True

    # DDP random sampler
    use_random_sampler: bool = True

    # if early stop when eval loss is not converging in the past early_stopping_stall_num evaluation point 
    early_stopping: bool = True
    early_stopping_stall_num: int = 5

    # limit num for saving ckpts, None for no limits. Used for full-parameter training to avoid exceeding disk quota.
    saving_limit: Union[None, int] = None

    # if dynamic padding 
    use_dynamic_padding: bool = True

    # interval of update per task train weight in selfpaced
    selfpaced_interval: int = 1
    # history length of sample valid loss used to fit the slope curve in selfpaced
    selfpaced_history_length: int = 100
    # the number of mini valid batches sampled at each interval
    selfpaced_sample_valid_num: int = 1
    # scale factor before softmax
    selfpaced_scale_factor: int = 50

    # ATTENTION_CLASSES = { "eager": Normal Attention, "flash_attention_2": FlashAttention2}
    attn_implementation: str = "flash_attention_2"

    # role markers, which are prompt template before each role: system, user and assistant
    # role_markers: {"system": "### System:\n", "user": "### Instruction:\n", "assistant": "### Response:\n"}
    role_markers: Union[None, dict] = None

    # legacy, leave them
    use_xformers: bool = True
    trust_remote_code: bool = True
    weight_by_num_documents: bool = True
    make_vocab_size_divisible_by: int = 32
    model_parallel_size: int = 1
    use_slow_tokenizer: bool = False
    world_size: int = 8

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}
