# @author Chaoyu Chen
# @date 2024/11/19
"""Distributed Inference by accelerate"""
import os
import json
import argparse
import queue
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from accelerate import PartialState
from accelerate.utils import gather_object

from infer_utils import (
    print_args,
    get_line_count,
    stream_jsonl,
    batch_stream_jsonl,
    flatten_batch_stream,
    write_jsonl,
)


def get_args():
    parser = argparse.ArgumentParser(description="Generation args.")
    parser.add_argument(
        "--model_path",
        type=str,
        help="huggingface model path",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        help="data file path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="output directory",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_return_sequences", type=int, default=20, help="pass1:20,pass10:20,pass100:100")
    parser.add_argument("--num_beams", type=int, default=1, help="beam1, beam3, beam5, beam7")
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument("--peft_path", type=str, default="", help="peft path：None")
    parser.add_argument("--eos_token", type=str, default=None, help="eos token")
    args = parser.parse_args()

    name = args.data_file.split("/")[-1].replace(".jsonl", "") + "-GEN"
    args.output_path = os.path.join(args.output_dir, f"{name}.jsonl")

    return args


def main():
    args = get_args()
    # Start up the distributed environment without needing the Accelerator.
    distributed_state = PartialState()

    # You can change the model to any LLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map=distributed_state.device, torch_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # Need to set the padding token to the eos token for generation
    tokenizer.pad_token = tokenizer.eos_token

    # get dataloader
    total_num = get_line_count(args.data_file)
    # You can change the batch size depending on your GPU RAM
    global_batch_size = args.batch_size * distributed_state.num_processes
    stream = stream_jsonl(args.data_file)
    dataloader = batch_stream_jsonl(stream, global_batch_size)

    # We set it to 8 since it is better for some hardware. More information here https://github.com/huggingface/tokenizers/issues/991
    pad_to_multiple_of = 8

    def handler():
        # streaming process
        for samples in dataloader:
            prompts = [sample['prompt'] for sample in samples]
            distributed_state.print(f"length of global batch: {len(prompts)}")

            # Split a global batch into batches with automatic padding so that the GPUs will have 1 batch with same size, and you can then gather the results.
            if len(prompts) == global_batch_size:
                formatted_prompts = [prompts[i: i + args.batch_size] for i in range(0, len(prompts), args.batch_size)]
            else:
                padded_prompts = prompts + [prompts[-1]] * (global_batch_size - len(prompts))
                formatted_prompts = [padded_prompts[i: i + args.batch_size] for i in
                                     range(0, len(padded_prompts), args.batch_size)]
            # distributed_state.print(f"formatted prompts {formatted_prompts}")

            # Apply padding on the left since we are doing generation
            padding_side_default = tokenizer.padding_side
            tokenizer.padding_side = "left"
            # Tokenize each batch
            tokenized_prompts = [
                tokenizer(formatted_prompt, padding=True, pad_to_multiple_of=pad_to_multiple_of, return_tensors="pt")
                for formatted_prompt in formatted_prompts
            ]
            # Put back the original padding behavior
            tokenizer.padding_side = padding_side_default

            completions_per_process = []
            with distributed_state.split_between_processes(tokenized_prompts) as batched_prompts:

                for batch in batched_prompts:
                    # Move the batch to the device
                    batch = batch.to(distributed_state.device)

                    # YOU WILL FILL YOUR CODE HERE!
                    # We generate the text, decode it and add it to the list completions_per_process
                    outputs = model.generate(
                        inputs=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_new_tokens=128
                    )

                    # generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    generated_text = tokenizer.batch_decode(outputs[:, batch["input_ids"].shape[1]:],
                                                            skip_special_tokens=True)
                    # print(f"index: {distributed_state.process_index}, {generated_text}, {len(generated_text)}")
                    completions_per_process.extend(generated_text)

            # We are gathering string, so we need to use gather_object.
            # If you need to gather tensors, you can use gather from accelerate.utils
            completions_gather = gather_object(completions_per_process)
            # Drop duplicates produced by apply_padding in split_between_processes
            completions = completions_gather[: len(prompts)]
            distributed_state.print(completions, len(completions))
            for sample, completion in zip(samples, completions):
                sample['generation'] = completion

            yield samples

    def save_results(output_queue: queue.Queue, output_path):
        with open(output_path, "w") as fp:
            while True:
                try:
                    item = output_queue.get(timeout=5)
                    if item is None:
                        break
                    fp.write(json.dumps(item, ensure_ascii=False) + "\n")
                    fp.flush()
                except queue.Empty:
                    continue

    # handel batch dataloader
    batch_res_stream = handler()
    res_stream = flatten_batch_stream(batch_res_stream)

    if distributed_state.is_main_process:
        output_queue = queue.Queue()
        save_thread = ThreadPoolExecutor(max_workers=16)
        save_future = save_thread.submit(save_results, output_queue, args.output_path)
        for res in res_stream:
            output_queue.put(res)
        output_queue.put(None)
        save_thread.shutdown(wait=True)
    else:
        list(res_stream)


if __name__ == "__main__":
    main()
