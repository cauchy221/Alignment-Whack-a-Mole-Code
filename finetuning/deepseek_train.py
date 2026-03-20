#!/usr/bin/env python3
"""
Finetune DeepSeek-V3.1 on paragraph-summary pairs via Tinker.

Launches a LoRA supervised finetuning job on the Tinker platform.  Training
data must first be converted to Tinker's chat JSONL format using
deepseek_convert.py.

Hyperparameters used in the paper (Appendix A.3):
  - LoRA rank: 32
  - Learning rate: 5e-4
  - Batch size: 16
  - Epochs: 3
  - Max sequence length: 2048
  - Renderer: deepseekv3_disable_thinking (disables DeepSeek's thinking mode)

Usage:
    # Step 1: Convert data
    python finetuning/deepseek_convert.py \
        --input_file data/example_book.json \
        --output_file train_messages.jsonl

    # Step 2: Train (chz uses key=value syntax, not --flags)
    python finetuning/deepseek_train.py \
        dataset=train_messages.jsonl \
        log_path=./logs/deepseek-mccarthy-epoch3

Requires:
    - pip install tinker tinker-cookbook
    - TINKER_API_KEY environment variable set.
      Sign up at https://auth.thinkingmachines.ai/sign-up and generate a key
      from https://tinker-console.thinkingmachines.ai.
"""

import asyncio
from datetime import datetime

import chz
from tinker_cookbook import cli_utils
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilder, ChatDatasetBuilderCommonConfig


@chz.chz
class CLIConfig:
    # Dataset
    dataset: str = "train_messages.jsonl"

    # Model
    model_name: str = "deepseek-ai/DeepSeek-V3.1"
    renderer_name: str = "deepseekv3_disable_thinking"
    load_checkpoint_path: str | None = None

    # Training hyperparameters
    learning_rate: float = 5e-4
    lr_schedule: str = "linear"
    num_epochs: int = 3
    lora_rank: int = 32
    batch_size: int = 16
    max_length: int = 2048

    # Logging and checkpointing
    log_path: str | None = None
    save_every: int = 1000
    eval_every: int = 0
    infrequent_eval_every: int = 0

    # Infrastructure
    base_url: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def _get_dataset_builder(
    dataset: str,
    model_name: str,
    renderer_name: str,
    max_length: int,
    batch_size: int,
) -> ChatDatasetBuilder:
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=max_length,
        batch_size=batch_size,
    )
    if dataset.endswith(".jsonl"):
        return FromConversationFileBuilder(
            common_config=common_config,
            file_path=dataset,
        )
    raise ValueError(f"Dataset must be a .jsonl file, got: {dataset}")


def cli_main(cli_config: CLIConfig):
    model_short = cli_config.model_name.replace("/", "-")
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"{model_short}-{cli_config.lora_rank}rank-"
        f"{cli_config.learning_rate}lr-{cli_config.batch_size}batch-{date_str}"
    )

    log_path = cli_config.log_path or f"/tmp/tinker-whackamole/{run_name}"
    wandb_name = cli_config.wandb_name or run_name

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    config = train.Config(
        log_path=log_path,
        model_name=cli_config.model_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        dataset_builder=_get_dataset_builder(
            cli_config.dataset,
            cli_config.model_name,
            cli_config.renderer_name,
            cli_config.max_length,
            cli_config.batch_size,
        ),
        evaluator_builders=[],
        infrequent_evaluator_builders=[],
        learning_rate=cli_config.learning_rate,
        lr_schedule=cli_config.lr_schedule,
        num_epochs=cli_config.num_epochs,
        base_url=cli_config.base_url,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        lora_rank=cli_config.lora_rank,
        save_every=cli_config.save_every,
        eval_every=cli_config.eval_every,
        infrequent_eval_every=cli_config.infrequent_eval_every,
    )
    asyncio.run(train.main(config))


if __name__ == "__main__":
    chz.nested_entrypoint(cli_main)
