import os
import time
import importlib.util
import openai
try:
    from vllm import LLM, SamplingParams
except ImportError:
    pass
from openai import OpenAI
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from prompts import icl_user_prompt, icl_ass_prompt


def _load_vllm_classes():
    if importlib.util.find_spec("vllm") is None:
        raise ImportError("backend='vllm' requires the optional vllm package to be installed.")
    from vllm import LLM, SamplingParams

    return LLM, SamplingParams


def llm_init(model_name, tensor_parallel_size=1, max_seq_len_to_capture=8192, max_tokens=4000, seed=0, temperature=0, frequency_penalty=0, backend="openai"):
    if backend == "vllm":
        LLM, SamplingParams = _load_vllm_classes()
        # client = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, max_seq_len_to_capture=max_seq_len_to_capture)
        client = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, max_model_len=max_seq_len_to_capture, gpu_memory_utilization=0.9)
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                         frequency_penalty=frequency_penalty)
        llm = partial(client.chat, sampling_params=sampling_params, use_tqdm=False)
    elif backend == "openai":
        # api_key = input("Enter OpenAI API key: ")
        # os.environ["OPENAI_API_KEY"] = api_key
        client = OpenAI()
        llm = partial(client.chat.completions.create, model=model_name, seed=seed, temperature=temperature, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unsupported LLM backend: {backend}")
    return llm


def get_outputs(outputs, model_name, backend="openai"):
    if backend == "vllm":
        return outputs[0].outputs[0].text
    else:
        # print("reasoning_content", outputs.choices[0].message.reasoning_content)
        # print("content", outputs.choices[0].message.content)
        return outputs.choices[0].message.content if outputs.choices and outputs.choices[0].message.content else ""


# v1
# def _build_select_user_query(user_query):
#     select_suffix = (
#         "\n\nFollow the select-then-answer format: begin with <think>, include exactly one "
#         "<select>...</select> block containing only comma-separated integer triplet "
#         "indices in the middle of the reasoning trace, close with </think>, and then "
#         "provide final answers as lines prefixed with \"ans:\"."
#     )
#     if "<select>" in user_query and "<think>" in user_query:
#         return user_query
#     return user_query + select_suffix

def _build_select_user_query(user_query):
    select_suffix = (
        "\nLet's think step by step. "
        "return formatted answers as a list, each prefixed with \"ans:\"."
    )
    # if "<select>" in user_query and "<think>" in user_query:
    #     return user_query
    return user_query + select_suffix


def llm_inf(llm, prompts, mode, model_name, backend="openai"):
    res = []
    use_select = 'select' in mode
    if use_select and 'icl' in mode:
        raise ValueError("llm_mode containing 'select' is mutually exclusive with 'icl'.")

    use_sys = 'sys' in mode or use_select
    if use_sys:
        conversation = [{"role": "system", "content": prompts['sys_query']}]
    else:
        conversation = []

    if 'icl' in mode and not use_select:
        conversation.append({"role": "user", "content": icl_user_prompt})
        conversation.append({"role": "assistant", "content": icl_ass_prompt})

    if use_sys:
        user_query = prompts['user_query']
        if use_select:
            user_query = _build_select_user_query(user_query)
        conversation.append({"role": "user", "content": user_query})
        outputs = get_outputs(llm(messages=conversation), model_name, backend)
        res.append(outputs)

    if 'sys_cot' in mode:
        if 'clear' in mode:
            conversation = []
            if use_select:
                conversation.append({"role": "system", "content": prompts['sys_query']})
        conversation.append({"role": "assistant", "content": outputs})
        conversation.append({"role": "user", "content": prompts['cot_query']})
        outputs = get_outputs(llm(messages=conversation), model_name, backend)
        res.append(outputs)
    elif "dc" in mode:
        if not res[0] or 'ans:' not in res[0].lower() or "ans: not available" in res[0].lower() or "ans: no information available" in res[0].lower():
            conversation.append({"role": "user", "content": prompts['cot_query']})
            outputs = get_outputs(llm(messages=conversation), model_name, backend)
            res[0] = outputs
        res.append("")
    else:
        res.append("")

    return res


def llm_inf_with_retry(llm, each_qa, llm_mode, model_name, max_retries, backend="openai"):
    retries = 0
    while retries < max_retries:
        try:
            return llm_inf(llm, each_qa, llm_mode, model_name, backend)
        except:
            wait_time = (2 ** retries) * 5  # Exponential backoff
            print(f"Rate limit error encountered. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
    raise Exception("Max retries exceeded. Please check your rate limits or try again later.")


def _llm_inf_one(llm, each_qa, llm_mode, model_name, max_retries=5, backend="openai"):
    if backend == "openai":
        return llm_inf_with_retry(llm, each_qa, llm_mode, model_name, max_retries, backend)
    return llm_inf(llm, each_qa, llm_mode, model_name, backend)


def llm_inf_all(llm, each_qa, llm_mode, model_name, max_retries=5, backend="openai", num_workers=1):
    """Run LLM inference for one QA item or a list of QA items.

    For a list input, OpenAI-compatible requests can be executed concurrently with
    a thread pool. Results are yielded in the same order as the input list so
    callers can append checkpoints sequentially and keep prefix-based resume
    behavior intact.
    """
    if isinstance(each_qa, dict):
        return _llm_inf_one(llm, each_qa, llm_mode, model_name, max_retries, backend)

    if num_workers <= 1 or backend != "openai":
        return (
            _llm_inf_one(llm, qa, llm_mode, model_name, max_retries, backend)
            for qa in each_qa
        )

    def _run_concurrent():
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            yield from executor.map(
                lambda qa: _llm_inf_one(llm, qa, llm_mode, model_name, max_retries, backend),
                each_qa,
            )

    return _run_concurrent()
