# -*- coding: utf-8 -*-

import torch
from datasets import load_dataset
from kaggle_secrets import UserSecretsClient
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer


user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("Hugging Face Access")

model_name = "meta-llama/Llama-3.2-3B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=secret_value_0,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    token=secret_value_0,
)

# Original base-model smoke test.
messages = [
    {
        "role": "user",
        "content": "what is love?",
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
)

response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print(response)


# Dataset loading and preparation.
dataset = load_dataset(
    "kisanVaani/agriculture-qa-english-only",
    token=secret_value_0,
)

print(dataset)
print(dataset["train"][0])

for i in range(10):
    print(dataset["train"][i])
    print("-" * 80)

print(dataset["train"].features)
print(dataset["train"].num_rows)

dataset = dataset["train"].train_test_split(
    test_size=0.1,
    seed=42,
)

print(dataset)


def format_example(example):
    return {
        "messages": [
            {
                "role": "user",
                "content": example["question"],
            },
            {
                "role": "assistant",
                "content": example["answers"],
            },
        ]
    }


formatted_dataset = dataset.map(
    format_example,
    remove_columns=["question", "answers"],
)

print(formatted_dataset["train"][0])


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

del model
torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    token=secret_value_0,
)

print(model.get_memory_footprint() / 1024**3, "GB")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = SFTConfig(
    output_dir="./llama-agriculture",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=250,
    save_strategy="steps",
    save_steps=250,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=False,
    bf16=True,
    gradient_checkpointing=True,
    use_liger_kernel=False,
    report_to="none",
    loss_type="nll",
)

eval_dataset = formatted_dataset["test"]

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset["train"],
    eval_dataset=formatted_dataset["test"],
    processing_class=tokenizer,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2,
        )
    ],
)

train_result = trainer.train()
