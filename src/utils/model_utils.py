from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from typing import Tuple

import logging


def load_tokenizer(data_args, model_args,
                  # padding_side
) -> PreTrainedTokenizer:
    # logging.info("Loading tokenizer")
    tokenizer =  AutoTokenizer.from_pretrained(model_args.pretrained_model_name_or_path)

    if not tokenizer.pad_token:
        if data_args.new_pad_token:
            tokenizer.padding_side = 'left'
            tokenizer.pad_token = data_args.new_pad_token,
            tokenizer.add_special_tokens({"pad_token": data_args.new_pad_token})
        else:
            tokenizer.padding_side = 'right'
            tokenizer.pad_token = tokenizer.eos_token
            
    return tokenizer

def load_model(model_args, use_cpu: bool=False) -> AutoModelForCausalLM:

    # Set torch_dtype and attn_implementation
    torch_dtype, attn_implementation = set_torch_dtype_and_attn_implementation()

    # QLora Config
    quantization_config = get_quantization_config(model_args)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.pretrained_model_name_or_path,
        trust_remote_code=True,
        quantization_config=quantization_config if not use_cpu else None,
        device_map="cpu" if use_cpu else "auto",
        attn_implementation=attn_implementation,
        # low_cpu_mem_usage=True if not use_cpu else False
    )

    return model


def get_model_tokenizer(
    data_args, model_args,
    *, 
    use_cpu: bool = False
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    tokenizer = load_tokenizer(data_args, model_args)
    model = load_model(model_args, use_cpu=True)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer



from transformers import BitsAndBytesConfig
def get_quantization_config(load_in_4bit: bool=False, 
                            load_in_8bit: bool=False,
                            bnb_4bit_compute_dtype=None,
                            bnb_4bit_quant_type: str="nf4",
                            bnb_4bit_use_double_quant: bool=False,
                            bnb_4bit_quant_storage: str="uint8"
                            
                            
) -> BitsAndBytesConfig | None:
    if load_in_4bit:
        # compute_dtype = torch.float16
        # if torch_dtype not in {"auto", None}:
        #     compute_dtype = getattr(torch, model_args.torch_dtype)
        

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_quant_storage=bnb_4bit_quant_storage,
        ).to_dict()
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        ).to_dict()
    else:
        quantization_config = None

    return quantization_config

import torch
def set_torch_dtype_and_attn_implementation():
    # Set torch dtype and attention implementation
    try:
        if torch.cuda.get_device_capability()[0] >= 8:
            # !pip install -qqq flash-attn
            torch_dtype = torch.bfloat16
            attn_implementation = "flash_attention_2"
        else:
            torch_dtype = torch.float16
            attn_implementation = "eager"
    except:
        torch_dtype = torch.float16
        attn_implementation = "eager"

    return torch_dtype, attn_implementation

def get_max_length(model):
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length



def find_target_modules(model) -> list[str]:
    # Initialize a Set to Store Unique Layers
    unique_layers = set()

    # Iterate Over All Named Modules in the Model
    for name, module in model.named_modules():
        # Check if the Module Type Contains 'Linear4bit'
        if "Linear4bit" in str(type(module)):
            # Extract the Type of the Layer
            layer_type = name.split('.')[-1]

            # Add the Layer Type to the Set of Unique Layers
            unique_layers.add(layer_type)

    # Return the Set of Unique Layers Converted to a List
    return list(unique_layers)


from peft import LoraConfig, PeftConfig
# def get_peft_config(model_args: ModelArguments) -> PeftConfig | None:
def get_peft_config(model_args) -> PeftConfig | None:
    if model_args.use_peft is False:
        return None

    peft_config = LoraConfig(
        r=model_args.lora.r,
        lora_alpha=model_args.lora.lora_alpha,
        lora_dropout=model_args.lora.lora_dropout,
        bias=model_args.lora.bias,
        task_type=model_args.lora.task_type,
        target_modules=list(model_args.lora.target_modules),
        modules_to_save=model_args.lora.modules_to_save,
    )

    return peft_config


# def get_peft_config(r, 
#                     lora_alpha, 
#                     target_modules, 
#                     lora_dropout, 
#                     bias, 
#                     task_type
# ):
#     """
#     Create Parameter-Efficient Fine-Tuning (PEFT) configuration for the model.

#     Args:
#         r (int): LoRA attention dimension.
#         lora_alpha (float): Alpha parameter for LoRA scaling.
#         target_modules (list): Names of the modules to apply LoRA to.
#         lora_dropout (float): Dropout probability for LoRA layers.
#         bias (str): Specifies if the bias parameters should be trained.
#         task_type (str): Type of task for PEFT (e.g., "CAUSAL_LM").

#     Returns:
#         LoraConfig: Configured LoRA settings for fine-tuning.
#     """
#     config = LoraConfig(
#         r=r,
#         lora_alpha=lora_alpha,
#         target_modules=target_modules,
#         lora_dropout=lora_dropout,
#         bias=bias,
#         task_type=task_type,
#     )
#     return config


# https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/supervised_finetuning.py
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def print_parameter_datatypes(model, logger=None):
    dtypes = dict()
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    
    total = 0
    for k, v in dtypes.items(): total += v

    for k, v in dtypes.items():

        if logger is None:
            print(f'type: {k} || num: {v} || {round(v/total, 3)}')
        else:
            logger.info(f'type: {k} || num: {v} || {round(v/total, 3)}')

def param_count(model):
    params = sum([p.numel() for p in model.parameters()])/1_000_000
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])/1_000_000
    print(f"Total params: {params:.2f}M, Trainable: {trainable_params:.2f}M")
    return params, trainable_params

from pathlib import Path
import os
from transformers.trainer_utils import get_last_checkpoint

def get_checkpoint(training_args) -> Path | None:
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint