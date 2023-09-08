import os
import argparse
from transformers import MODEL_MAPPING, SchedulerType

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--load_raw_dataset",
        action="store_true",
        help="If passed, will load raw dataset.",
    )
    parser.add_argument(
        "--load_hf_dataset",
        action="store_true",
        help="If passed, will load raw dataset.",
    )
    parser.add_argument(
        "--data_paths",
        type=str,
        default=None,
        help="Data path list.",
    )
    parser.add_argument(
        "--data_weights",
        type=str,
        default=None,
        help="Data weights.",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default=None,
        help="Data split.",
    )
    parser.add_argument(
        "--padding",
        action="store_true",
        help="use padding in preprocess.",
    )
    parser.add_argument(
        "--tokenize_mode",
        type=str,
        default='sft',
        choices=['pretrain', 'sft', 'coh'],
        help="training mode"
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default='sft',
        choices=['sst', 'sft'],
        help="training mode"
    )
    parser.add_argument(
        "--weighted_loss_mode",
        type=str,
        default=None,
        help="weighted loss mode.",
    )
    parser.add_argument(
        "--padding_mode",
        type=str,
        default='padding',
        choices=['padding', 'concat', 'pack'],
        help="padding mode"
    )
    parser.add_argument(
        "--shuffle_before_split",
        action="store_true",
        help="shuffle before split.",
    )
    parser.add_argument(
        "--use_random_sampler",
        action="store_true",
        help="use random sampler.",
    )
    parser.add_argument(
        "--use_weighted_loss",
        action="store_true",
        help="use weighted loss.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=-1,
        help="patience of early stopping.",
    )
    parser.add_argument(
        "--weight_by_num_documents",
        action="store_true",
        help="weight by num documents.",
    )
    parser.add_argument(
        "--make_vocab_size_divisible_by",
        type=int,
        default=128,
        help="Pad the vocab size to be divisible by this value. This is added for computational efficiency reasons.",
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default='gpt_neox',
        help="model type",
    )
    parser.add_argument(
        "--peft_type",
        type=str,
        default=None,
        help="peft type",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="How many bits to use when using qlora. Should be 4 or 8.",
    )
    parser.add_argument(
        "--use_xformers",
        action="store_true",
        help="use xformers in llama",
    )
    parser.add_argument(
        "--use_dynamic_padding",
        action="store_true",
        help="use xformers in llama",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--vocab_file",
        type=str,
        default=None,
        help="Vocab path",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default=None,
        help="Pretrained tokenizer type",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="If passed, will set trust_remote_code=True when calling from_pretrained.",
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="If passed, will set ignore_mismatched_sizes=True when calling from_pretrained.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_valid_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--total_train_batch_size",
        type=int,
        default=8,
        help="All batch size for the training dataloader. Equals to per_device_train_batch_size * world_size",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=8,
        help="Total world size (i.e number of gpus in cluster). Configured post-launch using distributed launcher",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=8,
        help="local rank passed from distributed launcher",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-7,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--custom_lr_scheduler_type",
        type=str,
        default=None,
        help="The scheduler type to use.",
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--num_update_steps_per_epoch",
        type=int,
        default=500,
        help="Number of update steps per epoch.",
    )
    parser.add_argument(
        "--resume_step",
        type=int,
        default=0,
        help="resume step in checkpoint.",
    )
    parser.add_argument(
        "--fp16_lm_cross_entropy",
        action="store_true",
        help="Move the cross entropy unreduced loss calculation for lm head to fp16.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="use bf16.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="use fp16.",
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Where to store the final model.")
    parser.add_argument(
        "--tensorboard_dir", 
        type=str, 
        default=None, 
        help="Where to store the tensorboard.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_types",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=None,
        help=(
            "hidden states size"
        ),
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=None,
        help=(
            "hidden states layers number"
        ),
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=None,
        help=(
            "vocab size"
        ),
    )
    parser.add_argument(
        "--total_model_param",
        type=int,
        default=None,
        help=(
            "total model parameters"
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="The number of sub-processes to use for the dataloader.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=0,
        help="Iteration to load checkpoint from in evaluate.py / generate.py. If None is provided, uses the latest iteration.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--no_keep_linebreaks",
        action="store_true",
        help="Do not keep line breaks when using TXT files.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder, including model checkpoint, optimizer and lr scheduler. path",
    )
    parser.add_argument(
        "--flash_attention",
        action="store_true",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--skip_atorch_autoacc_dryrun",
        action="store_true",
    )
    parser.add_argument(
        "--zero_opt_level",
        type=str,
        default="zero2",
        help="Model type to use if training from scratch.",
        choices=["zero2", "zero3", "fsdp"],
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--real_dataloader",
        action="store_true",
        help="Whether to use real full dataset.",
    )
    parser.add_argument(
        "--split_before_read",
        action="store_true",
        help="Whether to use real full dataset.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        help="tp size",
        required=True,
    )
    parser.add_argument(
        "--dp",
        type=int,
        help="dp size",
        required=True,
    )
    parser.add_argument(
        "--pipe_parallel_size",
        type=int,
        default=0,
        help="Number of pipeline parallel stages. Disable with 0.",
    )
    parser.add_argument(
        "--model_parallel_size",
        type=int,
        default=1,
        help="model parallelism. size",
    )
    parser.add_argument(
        "--train_iters",
        type=int,
        default=20000,
        help="Number of iterations to run for training.",
    )
    parser.add_argument(
        "--valid_iters",
        type=int,
        default=100,
        help="Number of iterations to run for evaluation validation/test for.",
    )
    parser.add_argument(
        "--valid_interval",
        type=int,
        default=1000,
        help="Interval between running evaluation on validation set.",
    )
    parser.add_argument(
        '--is_pipe_parallel',
        action='store_true',
        help="")
    parser.add_argument(
        "--log_interval",
        type=int,
        help="interval of logger",
        required=True,
    )
    parser.add_argument(
        '--checkpoint_activations',
        action='store_true',
        help="whether to use gradient checkpointing")
    parser.add_argument(
        "--max_grad_norm",
        type=int,
        help="max_grad_norm",
        required=True,
    )
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="steps",
        help="evaluation strategy",
        choices=["steps", "epoch", "steps,epoch"],
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="steps",
        help="save strategy",
        choices=["steps", "epoch", "steps,epoch"],
    )
    parser.add_argument(
        '--extra_save_by_epoch',
        action='store_true',
        help="whether to save extra checkpoint for per epoch"
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        help="num limitation of step strategy checkpoint file",
        required=True,
    )
    parser.add_argument(
        "--glm_mask",
        type=str,
        default="[gMASK]",
        help="Mask to use in glm",
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="loss",
        help="metric for best model checkpoint",
    )
    parser.add_argument(
        "--greater_is_better",
        type=str,
        default="false",
        help="whether the metric greater is better",
    )
    args = parser.parse_args()

    if os.getenv("OMPI_COMM_WORLD_LOCAL_RANK"):
        args.local_rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK"))
        print(args.local_rank)
    # Sanity checks
    if (
        args.data_paths is None
    ):
        raise ValueError("Need either a dataset name or a training/validation file.")

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args