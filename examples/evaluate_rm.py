import argparse
import math
import os
from collections import OrderedDict
from datetime import datetime
from tqdm import tqdm
import torch

from transformers.trainer import get_scheduler

from openrlhf.datasets import RewardDataset
from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.trainer import RewardModelTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer



def concatenated_inputs(tokenizer, chosen_ids, c_mask, reject_ids, r_mask):
    """Concatenate the chosen and rejected inputs into a single tensor.

    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """

    def pad_to_length(tensor, length, pad_value, dim=-1):
        if tensor.size(dim) >= length:
            return tensor
        else:
            pad_size = list(tensor.shape)
            pad_size[dim] = length - tensor.size(dim)
            # left pad
            return torch.cat(
                [pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device), tensor], dim=dim
            )

    max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
    inputs_ids = torch.cat(
        (
            pad_to_length(chosen_ids, max_length, tokenizer.pad_token_id),
            pad_to_length(reject_ids, max_length, tokenizer.pad_token_id),
        ),
        dim=0,
    )
    max_length = max(c_mask.shape[1], r_mask.shape[1])
    att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
    return inputs_ids, att_masks



def concatenated_forward(model, tokenizer, chosen_ids, c_mask, reject_ids, r_mask):
    """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

    We do this to avoid doing two forward passes, because it's faster for FSDP.
    """
    input_ids, att_masks = concatenated_inputs(tokenizer, chosen_ids, c_mask, reject_ids, r_mask)
    all_values, output = model(input_ids, attention_mask=att_masks, return_output=True)
    chosen_rewards = all_values[: chosen_ids.shape[0]]
    rejected_rewards = all_values[chosen_ids.shape[0] :]
    aux_loss = output.aux_loss if "aux_loss" in output else []
    return chosen_rewards, rejected_rewards, aux_loss


def evaluate(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model/config
    model = get_llm_for_sequence_regression(
        args.pretrain,
        "reward",
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=False),
        init_value_head=True,
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    strategy.print(model)


    # prepare for data and dataset
    _, eval_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=5000000,
        stopping_strategy="all_exhausted",
    )
    eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
    eval_dataset = RewardDataset(eval_data, tokenizer, args.max_len, strategy, input_template=args.input_template)

    eval_dataloader = strategy.setup_dataloader(
        eval_dataset, args.micro_train_batch_size, True, False, eval_dataset.collate_fn
    )

    # strategy prepare
    model = strategy.prepare(model)

    # batch_size here is micro_batch_size * 2
    # we use merged chosen + rejected response forward
    
    step_bar = tqdm(
        range(eval_dataloader.__len__()),
        disable=not strategy.is_rank_0(),
    )
    model.eval()
    with torch.no_grad():
        acc = 0
        rewards = []
        loss_sum = 0
        for chosen_ids, c_mask, reject_ids, r_mask, margin in eval_dataloader:
            chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
            c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
            reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
            r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
            margin = torch.tensor(margin).to(torch.cuda.current_device())

            chosen_reward, reject_reward, _ = concatenated_forward(
                model, tokenizer, chosen_ids, c_mask, reject_ids, r_mask
            )

            rewards += [chosen_reward.flatten(), reject_reward.flatten()]
            acc += (chosen_reward > reject_reward).float().mean().item()
            step_bar.update()

        acc_mean = acc / eval_dataloader.__len__()
        loss_mean = loss_sum / eval_dataloader.__len__()

        rewards = torch.cat(rewards).float()
        rewards = strategy.all_gather(rewards)
        reward_mean = torch.mean(rewards)
        reward_std = torch.std(rewards).clamp(min=1e-8)

        # save mean std
        strategy.print("Set reward mean std")
        unwrap_model = strategy._unwrap_model(model)
        unwrap_model.config.mean = reward_mean.item()
        unwrap_model.config.std = reward_std.item()

        bar_dict = {
            "eval_loss": loss_mean,
            "acc_mean": acc_mean,
            "reward_mean": reward_mean.item(),
            "reward_std": reward_std.item(),
        }
        logs = strategy.all_reduce(bar_dict)
        step_bar.set_postfix(logs)

        histgram = torch.histogram(rewards.cpu(), bins=10, range=(-10, 10), density=True) * 2
        strategy.print("histgram")
        strategy.print(histgram)
        
        print(bar_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="bigscience/bloomz-1b7")
    # parser.add_argument('--dataset', type=str, default='Anthropic/hh-rlhf')
    parser.add_argument("--dataset", type=str, default="Dahoas/full-hh-rlhf")
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_rm")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)  # 1000GB
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--micro_train_batch_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--loss", type=str, default="sigmoid")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--compute_fp32_loss", action="store_true", default=False)
    parser.add_argument("--margin_loss", action="store_true", default=False)
    parser.add_argument("--aux_loss_coef", type=float, default=0)
    parser.add_argument("--grad_accum_dtype", type=str, default=None)
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--input_template", type=str, default="[INST] {} [/INST] ")  # default="Human:\n{}\nAssistant:\n")
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # custom dataset key name
    parser.add_argument("--prompt_key", type=str, default=None)
    parser.add_argument("--chosen_key", type=str, default=None)
    parser.add_argument("--rejected_key", type=str, default=None)

    # wandb pamameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_rm")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="rm_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    args = parser.parse_args()
    evaluate(args)

