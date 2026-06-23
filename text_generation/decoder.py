import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .prompt import (
    SYSTEM_PROMPT, COLUMN_DESCRIPTION,
    build_starting_reasoning_prompt,
    build_arrival_reasoning_prompt,
    build_caption_generation_prompt,
    build_simple_prompt,
)

DEFAULT_MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def load_llm(model_path=DEFAULT_MODEL_PATH, device='cuda:0'):
    """加载LLM和分词器"""
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded.")
    return model, tokenizer


def _llm_call(model, tokenizer, prompt, max_new_tokens=64,
              temperature=0.7, top_p=0.9):
    """单次LLM调用"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    terminators = [tokenizer.eos_token_id]
    eot_token = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_token != tokenizer.eos_token_id:
        terminators.append(eot_token)

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=True, temperature=temperature, top_p=top_p,
    )

    generated_ids = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def generate_caption_multi_stage(model, tokenizer,
                                  trajectory_info_text,
                                  pickup_poi_info, dropoff_poi_info,
                                  pickup_hour, pickup_week,
                                  dropoff_hour, dropoff_week,
                                  max_new_tokens=128,
                                  temperature=0.7, top_p=0.9):
    """
    多阶段推理：三次独立LLM调用。

    Stage 1 (Starting): 起点POI + 时间 → 旅行者身份/出发原因
    Stage 2 (Arrival):  终点POI + 时间 → 到达意图
    Stage 3 (Caption):  聚合 Stage1+2 + 原始属性 → 最终描述
    """
    # Stage 1: Starting semantic reasoning
    prompt_s = build_starting_reasoning_prompt(
        pickup_poi_info, pickup_hour, pickup_week
    )
    ans_s = _llm_call(model, tokenizer, prompt_s, max_new_tokens=64,
                      temperature=temperature, top_p=top_p)

    # Stage 2: Arrival semantic reasoning (independent, could run in parallel)
    prompt_a = build_arrival_reasoning_prompt(
        dropoff_poi_info, dropoff_hour, dropoff_week
    )
    ans_a = _llm_call(model, tokenizer, prompt_a, max_new_tokens=64,
                      temperature=temperature, top_p=top_p)

    # Stage 3: Trajectory caption generation
    prompt_tr = build_caption_generation_prompt(
        trajectory_info_text, pickup_poi_info, dropoff_poi_info, ans_s, ans_a
    )
    caption = _llm_call(model, tokenizer, prompt_tr, max_new_tokens=max_new_tokens,
                        temperature=temperature, top_p=top_p)

    return caption


def generate_caption_simple(model, tokenizer,
                             trajectory_info_text, semantic_info_text,
                             max_new_tokens=128,
                             temperature=0.7, top_p=0.9):
    """
    单阶段直接生成（消融实验 w/o multi-stage reasoning）。
    不经过Stage 1/2的中间推理步骤。
    """
    prompt = build_simple_prompt(trajectory_info_text, semantic_info_text)
    return _llm_call(model, tokenizer, prompt, max_new_tokens=max_new_tokens,
                     temperature=temperature, top_p=top_p)
