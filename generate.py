from src.utils.model_utils import get_model_tokenizer
from typing import List, Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Pipeline,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from transformers import BitsAndBytesConfig
from src.utils.model_utils import set_torch_dtype_and_attn_implementation, get_quantization_config


def get_quantization_config(**model_args                          
) -> BitsAndBytesConfig | None:
    if model_args['load_in_4bit']:
        # compute_dtype = torch.float16
        # if torch_dtype not in {"auto", None}:
        #     compute_dtype = getattr(torch, model_args.torch_dtype)
        torch_dtype, attn_implementation = set_torch_dtype_and_attn_implementation()

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            # bnb_4bit_compute_dtype=model_args['bnb_4bit_compute_dtype'],
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type=model_args['bnb_4bit_quant_type'],
            bnb_4bit_use_double_quant=model_args['bnb_4bit_use_double_quant'],
            bnb_4bit_quant_storage=model_args['bnb_4bit_quant_storage'],
        ).to_dict()
    elif model_args['load_in_8bit']:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            # bnb_8bit_compute_dtype='float16',
            # bnb_8bit_use_double_quant=model_args['bnb_4bit_use_double_quant'],
            # bnb_8bit_quant_type=model_args['bnb_4bit_quant_type'],
        ).to_dict()
    else:
        quantization_config = None

    return quantization_config


def load_model_for_generate(
    use_cpu: bool=False,
    **model_args,
) -> PreTrainedModel:
    
    torch_dtype, attn_implementation = set_torch_dtype_and_attn_implementation()
    # quantization_config = get_quantization_config(**model_args)

    # print("torch_dtype:", torch_dtype)
    # print("attn_implementation:", attn_implementation)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args['pretrained_model_name_or_path'],
        trust_remote_code=True,
        quantization_config=get_quantization_config(**model_args) if not use_cpu else None,
        device_map="cpu" if use_cpu else "auto",
        torch_dtype="float32",
        attn_implementation=attn_implementation,
        # low_cpu_mem_usage=True if not use_cpu else False
    )

    return model

def load_tokenizer_for_generate(
    **model_args
) -> PreTrainedTokenizer:
    
    tokenizer = AutoTokenizer.from_pretrained(model_args['tokenizer_name_or_path'])

    return tokenizer

def prepare_prompt(**prompt_kwargs):

    if prompt_kwargs['use_no_keys']:
        prompt = prompt_kwargs['input_text']
        return prompt
        
    instruction = f"{prompt_kwargs['instruction_key']}\n{prompt_kwargs['instruction_text']}"

    examples = None

    if prompt_kwargs['use_examples']:
        examples = f"{prompt_kwargs['examples_key']}\n{prompt_kwargs['examples_text']}"
        
    input = f"{prompt_kwargs['input_key']}\n{prompt_kwargs['input_text']}"

    context = None
    if prompt_kwargs['use_context']:
        context = f"{prompt_kwargs['context_key']}\n{prompt_kwargs['context_text']}"

    response_key = f"{prompt_kwargs['response_key']}\n"

    parts = [part for part in [instruction, examples, context, input, response_key] if part]

    prompt = "\n\n".join(parts)
    
    return prompt

def postprocess(
        prompt: str,
        tokenizer, 
        response: str, 
        # response_key: str="### Answer:", 
        # end_key: str="<|im_end|>",
        return_full_text: bool=False, 
        **kwargs,
    ):
    import re
    decoded = None
    fully_decoded = response
    # pattern = r"#+\s*Answer:\s*(.+?)#+\s*End"
    
    if not kwargs['end_key']:
        kwargs['end_key'] = tokenizer.eos_token
    
    pattern = r".*?{}\s*(.+?){}".format(kwargs['response_key'], kwargs['end_key'])
    m = re.search(pattern, fully_decoded, flags=re.DOTALL)
    # print('m:', m.group(0))
    if m:
        decoded = m.group(1).strip()
    else:
        # The model might not generate the  kwargs['end_key'] sequence before reaching the max tokens.  In this case,
        # return everything after kwargs['response_key'].
        pattern = r".*?{}\s*(.+)".format(kwargs['response_key'])
        m = re.search(pattern, fully_decoded, flags=re.DOTALL)
        if m:
            decoded = m.group(1).strip()
        else:
            # logger.warn(f"Failed to find response in:\n{fully_decoded}")
            # pass
            print(f"Failed to find response in:\n{fully_decoded}")
            

    if return_full_text:
        decoded = f"{prompt}{decoded}"

    return decoded


def generate_response(use_cpu: bool=False, return_full_text=False, do_postprocess=True, **generate_kwargs):
    model = load_model_for_generate(use_cpu=use_cpu, **generate_kwargs)
    # print(model.config)
    tokenizer = load_tokenizer_for_generate(**generate_kwargs)

    # print(tokenizer)

    prompt = prepare_prompt(**generate_kwargs)

    # prompts = [prompt] * 2
    # print('prompt:', prompt)

    # input = sample['text']
    inputs = tokenizer(prompt, return_tensors='pt', padding=False, truncation=True)
    input_ids = inputs.input_ids

    input_ids = input_ids.to(model.device)
    
    output_ids = model.generate(input_ids, 
                                max_new_tokens=generate_kwargs['max_new_tokens'], 
                                pad_token_id=tokenizer.pad_token_id)

    # print('output_ids:', output_ids)
    # print('output_ids[[0]]:', output_ids[[0]])
    
    output = tokenizer.decode(output_ids[0], 
                              skip_special_tokens=generate_kwargs['skip_special_tokens'])
    # print('answer:', answer)
    if do_postprocess:
        # outputs = [postprocess(prompt, tokenizer, output, return_full_text, **generate_kwargs) for (prompt, output) in zip(prompts, outputs)]

        output = postprocess(prompt, tokenizer, output, return_full_text, **generate_kwargs)
    
    return output

import torch

def get_device():
    """
    Returns the available device (CUDA if available, otherwise CPU).

    Returns:
        torch.device: The device to use for computations.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Load generation config.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the YAML config file for generating.")
    return parser.parse_args()

if __name__ == "__main__":

    from omegaconf import OmegaConf
    args = parse_args()

    # Load the generation config file
    config = OmegaConf.load(args.config_path)

    model_kwargs = config.model
    prompt_kwargs = config.prompt
    generate_kwargs = config.generate

    kwargs = {**model_kwargs, **prompt_kwargs, **generate_kwargs}

    result = generate_response(use_cpu=False, return_full_text=True, do_postprocess=False, **kwargs)
    print(result)