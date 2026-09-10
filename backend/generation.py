import json
from threading import Thread

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from .config import Base_Model_ID, Use_8B, bnb_Config, checkpoint_path, get_huggingface_secret


secret_value = get_huggingface_secret()

tokenizer = AutoTokenizer.from_pretrained(
    Base_Model_ID,
    token=secret_value,
)

tokenizer.pad_token = tokenizer.eos_token


if Use_8B:
    base_model = AutoModelForCausalLM.from_pretrained(
        Base_Model_ID,
        token=secret_value,
        device_map="auto",
        quantization_config=bnb_Config,
    )
    print("Foundational Model Loaded...!")
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        Base_Model_ID,
        quantization_config=bnb_Config,
        device_map="auto",
        token=secret_value,
    )

    base_model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        is_trainable=False,
    )
    base_model.eval()
    print("Fine Tuned model loaded...!")


def stream_model_generation(prompt_text, max_new_tokens=512):
    """
    Stream generated text from the model token-by-token.
    """

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
    ).to(base_model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.3,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
        "pad_token_id": tokenizer.eos_token_id,
    }

    def generate():
        with torch.no_grad():
            base_model.generate(**generation_kwargs)

    thread = Thread(target=generate)
    thread.start()

    try:
        for text in streamer:
            yield text
    finally:
        thread.join()


def generate_final_answer_stream(query: str, tool_result: dict):
    from .prompts import system_prompt

    tool_result_text = json.dumps(
        tool_result,
        ensure_ascii=False,
        indent=2,
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt.format(tool_result=tool_result_text),
        },
        {
            "role": "user",
            "content": query,
        },
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    yield from stream_model_generation(
        prompt_text,
        max_new_tokens=1024,
    )


def generate_direct_answer_stream(query: str):
    messages = [
        {
            "role": "system",
            "content": """You are Project Kisan, a friendly and intelligent AI assistant.

Answer the user's message naturally and conversationally.

For casual conversation, greetings, general explanations,
and questions that do not require external retrieval, answer directly.

Do not mention tools, retrieval, prompts, or internal processing.

Do not fabricate specific facts.

Answer in the same language as the user.
""",
        },
        {
            "role": "user",
            "content": query,
        },
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    yield from stream_model_generation(
        prompt_text,
        max_new_tokens=1024,
    )
