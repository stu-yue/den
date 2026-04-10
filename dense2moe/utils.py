from collections import defaultdict
import os
import torch
import transformers 
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPTNeoXTokenizerFast,
    LlamaTokenizer,
    LlamaTokenizerFast,
)

# For OLMoE
DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|system|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|user|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{% if not loop.last %}"
    "{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}"
    "{% else %}"
    "{{ '<|assistant|>\n'  + message['content'] + eos_token }}"
    "{% endif %}"
    "{% endif %}"
    "{% if loop.last and add_generation_prompt %}"
    "{{ '<|assistant|>\n' }}"
    "{% endif %}"
    "{% endfor %}"
)


def load_hf_lm(
        model_name_or_path,
        revision=None,
        device_map="auto", 
        torch_dtype="auto",
        convert_to_half=True,
        token=os.getenv("HF_TOKEN", None),
    ):

    # Loading OLMo models from HF requires `trust_remote_code=True`.
    # TODO: Implement this via command-line flag rather than hardcoded list.
    trusted_models = ["allenai/OLMo-7B", "allenai/OLMo-7B-Twin-2T", "allenai/OLMo-1B"]
    if model_name_or_path in trusted_models:
        trust_remote_code = True
    else:
        trust_remote_code = False

    if device_map:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            revision=revision,
            device_map=device_map,
            torch_dtype=torch_dtype,
            token=token,
            trust_remote_code=trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            revision=revision,
            torch_dtype=torch_dtype,
            token=token,
            trust_remote_code=trust_remote_code,
        )
        if torch.cuda.is_available():
            model = model.cuda()
    if convert_to_half:
        model = model.half()
    model.eval()
    return model

def load_hf_tokenizer(
        model_name_or_path, 
        revision=None,
        use_fast_tokenizer=True,
        padding_side="left",
        token=os.getenv("HF_TOKEN", None),
        add_bos=True,
    ):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, token=token, revision=revision)
        except:
            # some tokenizers (e.g., GPTNeoXTokenizer) don't have the slow or fast version, so we just roll back to the default one
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=token, revision=revision)
        # set padding side to left for batch generation
        tokenizer.padding_side = padding_side
        # set pad token to eos token if pad token is not set (as is the case for llama models)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if tokenizer.chat_template is None:
            tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
        # no default pad token for llama!
        # here we add all special tokens again, because the default ones are not in the special_tokens_map
        # only add if the pad token is not present already, or if the current one is set to eos_token_id.
        if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
            if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
                num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
                assert num_added_tokens in [
                    0,
                    1,
                ], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
            elif isinstance(tokenizer, GPTNeoXTokenizerFast):
                # OLMo newer models use this tokenizer
                if tokenizer.bos_token is None:
                    tokenizer.bos_token = tokenizer.eos_token
                    assert (
                        add_bos
                    ), "For OLMo with GPTNeoX, you must add bos token to the beginning of the input sequence."
                # else, pythia / other models
                else:
                    num_added_tokens = tokenizer.add_special_tokens(
                        {
                            "pad_token": "<pad>",
                        }
                    )
                    assert (
                        num_added_tokens <= 1
                    ), "GPTNeoXTokenizer should only add one special token - the pad_token (or no tokens if already set in SFT)."
            # NOTE: (Costa) I just commented the `OPTForCausalLM` because we are not likely to use it.
            # elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
            #     num_added_tokens = tokenizer.add_special_tokens({"unk_token": "<unk>"})
            elif isinstance(tokenizer, transformers.PreTrainedTokenizerFast):
                num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
                assert num_added_tokens == 1, "We detected no padding token but add_special_tokens did not add one."

        if add_bos:
            if tokenizer.chat_template.startswith("{{ bos_token }}") or (
                tokenizer.bos_token is not None and tokenizer.chat_template.startswith(tokenizer.bos_token)
            ):
                raise ValueError(
                    "You specified add_bos=True, but the chat template already has a bos_token at the beginning."
                )
            # also add bos in the chat template if not already there
            tokenizer.chat_template = "{{ bos_token }}" + tokenizer.chat_template
        return tokenizer

def load_hf_lm_and_tokenizer(
        model_name_or_path, 
        revision=None,
        device_map="auto", 
        torch_dtype="auto",
        convert_to_half=True,
        padding_side="left",
        use_fast_tokenizer=True,
        token=os.getenv("HF_TOKEN", None),
    ):
        tokenizer = load_hf_tokenizer(
            model_name_or_path=model_name_or_path,
            revision=revision,
            use_fast_tokenizer=use_fast_tokenizer,
            padding_side=padding_side,
            token=token,
        )
        model = load_hf_lm(
            model_name_or_path=model_name_or_path,
            revision=revision,
            device_map=device_map,
            torch_dtype=torch_dtype,
            convert_to_half=convert_to_half,
            token=token,
        )
        from transformers import GPTNeoXForCausalLM, OPTForCausalLM
        if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
            tokenizer.model_max_length = model.config.max_position_embeddings
            print("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(model.config.max_position_embeddings))
        return model, tokenizer


def get_formatted_input_and_target(messages, tokenizer, IGNORE_TOKEN_ID=-100):
    input_ids = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        return_tensors="pt",
        padding=False,
        add_generation_prompt=False,
    )
    labels = input_ids.clone()
    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            # we calculate the start index of this non-assistant message
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer.apply_chat_template(
                    conversation=messages[:message_idx],  # here marks the end of the previous messages
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    add_generation_prompt=False,
                ).shape[1]
            # next, we calculate the end index of this non-assistant message
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                # for intermediate messages that follow with an assistant message, we need to
                # set `add_generation_prompt=True` to avoid the assistant generation prefix being included in the loss
                # (e.g., `<|assistant|>`)
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    add_generation_prompt=True,
                ).shape[1]
            else:
                # for the last message or the message that doesn't follow with an assistant message,
                # we don't need to add the assistant generation prefix
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    add_generation_prompt=False,
                ).shape[1]
            # set the label to -100 for the non-assistant part
            labels[:, message_start_idx:message_end_idx] = IGNORE_TOKEN_ID
    
    attention_mask = torch.ones_like(input_ids)
    return input_ids, labels



def find_device_mismatches(model, name="Model"):
    """
    Find and report device mismatches in the model
    
    Args:
        model: PyTorch model
        name: Model name
    
    Returns:
        dict: Dictionary of devices and their parameters
    """
    device_params = defaultdict(list)
    
    for param_name, param in model.named_parameters():
        device_params[param.device].append(param_name)
    
    if len(device_params) > 1:
        print(f"\n⚠️  DEVICE MISMATCHES FOUND in {name}:")
        for device, params in sorted(device_params.items()):
            print(f"\n  {device} ({len(params)} parameters):")
            for param_name in params[:10]:  # Show first 10
                print(f"    - {param_name}")
            if len(params) > 10:
                print(f"    ... and {len(params) - 10} more")
    else:
        print(f"\n✅ No device mismatches found in {name}")
        # Print device information even when there are no mismatches
        for device, params in sorted(device_params.items()):
            print(f"\n  {device} ({len(params)} parameters):")
            for param_name in params[:10]:  # Show first 10
                print(f"    - {param_name}")
            if len(params) > 10:
                print(f"    ... and {len(params) - 10} more")
    
    return dict(device_params)






def get_cumsum_expert_subset_by_stats(all_expt_stats, threshold):
    """Find the minimum expert subset s.t. cumsum >= threshold
    Args:
        all_expt_stats  : Dict[List]
        threshold       : float in [0, 1]
    """
    expert_subsets = {
        layer_idx: []
        for layer_idx in all_expt_stats.keys()
    }
    
    for layer_idx in all_expt_stats.keys():
        arr = all_expt_stats[layer_idx]     # list, expert scores in this layer
        expert_subsets[layer_idx] = find_minimal_subset_by_thresh(arr, threshold)

    return expert_subsets



def find_minimal_subset_by_thresh(arr, threshold):
    """Find the minimal subset s.t. cumsum >= threshold
    Args:
        arr             : List of expert statistics
        threshold       : float
    """
    if threshold > 0.9995:
        threshold = 0.9995
    
    arr = np.array(arr)
    sorted_indices = np.argsort(arr)[::-1]   # sort indices
    sorted_values = arr[sorted_indices]      # sort values
    cumsum = np.cumsum(sorted_values)                   # compute cumulative sum
    idx = np.argmax(cumsum >= threshold)
    
    return sorted_indices[: idx + 1].tolist()




