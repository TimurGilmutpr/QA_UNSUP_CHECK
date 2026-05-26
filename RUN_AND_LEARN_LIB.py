import torch
import time
import math
import numpy as np
from typing import Dict, List, Optional
from trl import SFTTrainer, SFTConfig

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)

from peft import LoraConfig, get_peft_model
from torch.optim import AdamW


def load_model(
    model_name: str,
    device: str = "cuda",               # "cuda" или "cpu"
    quantization: Optional[str] = None, # None | "4bit" | "8bit"
    torch_dtype=torch.bfloat16,
    token = "F"
):
    """
    Загружает модель с возможностью квантования.
    """

    if device == "cpu":
        torch_dtype = torch.float32

    quant_config = None

    if quantization == "4bit":
        quant_config = BitsAndBytesConfig(load_in_4bit=True)
    elif quantization == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch_dtype,
        quantization_config=quant_config,
        token = token,
        
    )

    if device == "cpu":
        model.to("cpu")

    return model, tokenizer



def finetune_lora(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    output_dir: str,
    r: int = 32,
    alpha: int = 64,
    dropout: float = 0.05,
    epochs: int = 40,
    lr: float = 2e-4,
    optim: str = "paged_adamw_8bit"):
    """
    LoRA дообучение (PEFT).
    """

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        lr_scheduler_type="cosine",
        learning_rate=lr,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to="none",
        optim=optim,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class = tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config,
        eval_dataset=val_dataset
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model



import gc

def cleanup_cuda():
    """Очистка Python GC и CUDA кэшей. Используйте после del model/tokenizer/..."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
