import json
import re

import torch

from .generation import base_model, tokenizer
from .prompts import AGENT_SYSTEM_PROMPT
from .rag import rag_search
from .web_search import web_search_tool


AVAILABLE_TOOLS = {
    "rag_search": rag_search,
    "web_search_tool": web_search_tool,
}


tools = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "Search Project Kisan's internal "
                "agricultural knowledge base."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search_tool",
            "description": (
                "Search the web for current, missing, "
                "or externally verifiable information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


def generate_tool_call(query: str):
    messages = [
        {
            "role": "system",
            "content": AGENT_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": query,
        },
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
    ).to(base_model.device)

    with torch.no_grad():
        output = base_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    generated_tokens = output[0][
        inputs["input_ids"].shape[1]:
    ]

    raw_output = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=False,
    )

    tool_call = extract_tool_call(raw_output)

    return normalize_tool_call(tool_call)


def extract_tool_call(raw_output):
    """
    Extract a tool call from the model's generated text.
    """

    raw_output = raw_output.replace("<|eot_id|>", "").strip()

    match = re.search(r"\{.*\}", raw_output, re.DOTALL)

    if not match:
        return None

    json_text = match.group(0)

    json_text = json_text.replace(
        '"parameters)"',
        '"parameters":',
    )

    try:
        return json.loads(json_text)

    except json.JSONDecodeError as e:
        print("JSON parsing failed:")
        print(json_text)
        print("\nError:", e)
        return None


def execute_native_tool(tool_call):
    if not tool_call:
        return {
            "status": "invalid_tool_call",
            "results": [],
        }

    tool_name = tool_call.get("name")

    if not tool_name:
        function_field = tool_call.get("function")

        if isinstance(function_field, str):
            tool_name = function_field

        elif isinstance(function_field, dict):
            tool_name = function_field.get("name")

    parameters = tool_call.get(
        "parameters",
        {},
    )

    if not isinstance(parameters, dict):
        parameters = {}

    if tool_name not in AVAILABLE_TOOLS:
        return {
            "status": "unknown_tool",
            "tool": tool_name,
            "results": [],
        }

    query = parameters.get("query")

    if not query:
        return {
            "status": "invalid_query",
            "results": [],
        }

    try:
        return AVAILABLE_TOOLS[tool_name](query)

    except Exception as e:
        return {
            "status": "tool_execution_failed",
            "results": [],
            "error": str(e),
        }


def normalize_tool_call(tool_call):
    if not tool_call:
        return None

    if tool_call.get("name"):
        return {
            "name": tool_call["name"],
            "parameters": tool_call.get(
                "parameters",
                {},
            ),
        }

    function_field = tool_call.get("function")

    if isinstance(function_field, str):
        return {
            "name": function_field,
            "parameters": tool_call.get(
                "parameters",
                {},
            ),
        }

    if isinstance(function_field, dict):
        return {
            "name": function_field.get("name"),
            "parameters": (
                tool_call.get("parameters")
                or function_field.get(
                    "arguments",
                    {},
                )
            ),
        }

    return None
