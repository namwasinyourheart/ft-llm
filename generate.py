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
        

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model_args['bnb_4bit_compute_dtype'],
            bnb_4bit_quant_type=model_args['bnb_4bit_quant_type'],
            bnb_4bit_use_double_quant=model_args['bnb_4bit_use_double_quant'],
            bnb_4bit_quant_storage=model_args['bnb_4bit_quant_storage'],
        ).to_dict()
    elif model_args['load_in_8bit']:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
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

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args['pretrained_model_name_or_path'],
        trust_remote_code=True,
        quantization_config=get_quantization_config(**model_args) if not use_cpu else None,
        device_map="cpu" if use_cpu else "auto",
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
    instruction = f"{prompt_kwargs['instruction_key']}\n{prompt_kwargs['instruction_text']}"

    examples = None
    if prompt_kwargs['use_fewshot']:
        examples = f"{prompt_kwargs['examples_key']}\n{prompt_kwargs['examples_text']}"
        
    input = f"{prompt_kwargs['input_key']}\n{prompt_kwargs['input_text']}"

    context = None
    if prompt_kwargs['use_context']:
        context = f"{prompt_kwargs['context_key']}\n{prompt_kwargs['context_text']}"

    response_key = prompt_kwargs['response_key']

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
        # return everything after "### Response:".
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
    tokenizer = load_tokenizer_for_generate(**generate_kwargs)

    # print(tokenizer)

    prompt = prepare_prompt(**generate_kwargs)

    # prompts = [prompt] * 2
    prompts = prompt

    # print('prompt:', prompt)

    # input = sample['text']
    inputs = tokenizer(prompts, return_tensors='pt', padding=False, truncation=True)
    input_ids = inputs.input_ids

    device = get_device()
    input_ids = input_ids.to(device)
    
    output_ids = model.generate(input_ids, 
                                max_new_tokens=generate_kwargs['max_new_tokens'], 
                                pad_token_id=tokenizer.pad_token_id)

    # print('output_ids:', output_ids)
    # print('output_ids[[0]]:', output_ids[[0]])
    
    outputs = tokenizer.batch_decode(output_ids, 
                              skip_special_tokens=generate_kwargs['skip_special_tokens'])
    # print('answer:', answer)
    if do_postprocess:
        outputs = [postprocess(prompt, tokenizer, output, return_full_text, **generate_kwargs) for (prompt, output) in zip(prompts, outputs)]
    
    return outputs

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

if __name__ == "__main__":

    # import argparse
    # parser = argparse.ArgumentParser(description='Generate an answer for a given question.')

    # # parser.add_argument(
    # #     '--model_name_or_path',
    # #     type=str,
    # #     required=True,
    # #     help='The path to model.'
    # # )
    # parser.add_argument(
    #     '--question',
    #     type=str,
    #     required=True,
    #     help='The question to generate an answer for.'
    # )
    # args = parser.parse_args()


    model_zoo = [
        'bigcode/starcoderbase-1b',
        'bigcode/starcoder2-3b',
        'deepseek-ai/deepseek-coder-1.3b-base',
        'deepseek-ai/deepseek-coder-1.3b-instruct',
        'codellama/CodeLlama-7b-hf',
        'Qwen/Qwen2.5-Coder-1.5B-Instruct',
        'Qwen/Qwen2.5-Coder-1.5B',
        'Qwen/Qwen2.5-Coder-0.5B-Instruct',
        'google/codegemma-2b',
        'HuggingFaceTB/SmolLM2-360M-Instruct',
        'HuggingFaceTB/SmolLM2-1.7B-Instruct',
        'HuggingFaceTB/SmolLM-135M'
    
]


    model_kwargs = {
        "pretrained_model_name_or_path": "meta-llama/Llama-3.2-3B-Instruct",
        "tokenizer_name_or_path": "meta-llama/Llama-3.2-3B-Instruct",
        "load_in_4bit": True,
        "load_in_8bit": False,
        "bnb_4bit_compute_dtype": None,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": False,
        "bnb_4bit_quant_storage": "uint8",
}
    prompt_kwargs = {
        'context_key': "### Context",
        'context_text': "CREATE TABLE city (Official_Name VARCHAR, Status VARCHAR, Population VARCHAR)",
        
        'instruction_key': "### Instruction:",
        'instruction_text': """You are an expert in data analysis using SQL. Given a question and context, your task is to write SQL query to find answer for the question .""",
        
        'input_key': '### Question:', 
        'input_text': "List the official name and status of the city with the largest population.",
        
        'response_key': "### SQL:",
        'end_key': None
    }

    absa_prompt_kwargs = {
        'use_fewshot': False,
        'use_context': True,
        'context_key': "### Categories:",
        'context_text': "['FACILITIES#CLEANLINESS', 'FACILITIES#COMFORT', 'FACILITIES#DESIGN&FEATURES', 'FACILITIES#GENERAL', 'FACILITIES#MISCELLANEOUS', 'FACILITIES#PRICES', 'FACILITIES#QUALITY', 'FOOD&DRINKS#MISCELLANEOUS', 'FOOD&DRINKS#PRICES', 'FOOD&DRINKS#QUALITY', 'FOOD&DRINKS#STYLE&OPTIONS', 'HOTEL#CLEANLINESS', 'HOTEL#COMFORT', 'HOTEL#DESIGN&FEATURES', 'HOTEL#GENERAL', 'HOTEL#MISCELLANEOUS', 'HOTEL#PRICES', 'HOTEL#QUALITY', 'LOCATION#GENERAL', 'ROOMS#CLEANLINESS', 'ROOMS#COMFORT', 'ROOMS#DESIGN&FEATURES', 'ROOMS#GENERAL', 'ROOMS#MISCELLANEOUS', 'ROOMS#PRICES', 'ROOMS#QUALITY', 'ROOM_AMENITIES#CLEANLINESS', 'ROOM_AMENITIES#COMFORT', 'ROOM_AMENITIES#DESIGN&FEATURES', 'ROOM_AMENITIES#GENERAL', 'ROOM_AMENITIES#MISCELLANEOUS', 'ROOM_AMENITIES#PRICES', 'ROOM_AMENITIES#QUALITY', 'SERVICE#GENERAL']",
        
        'instruction_key': "### Instruction:",
        'instruction_text': "You are an aspect-based sentiment analyzer in Vietnamese. Given a text of user review and a list of available aspect categories, your task is to define all sentiment elements with their corresponding aspect categories and sentiment polarity.",
        # 'instruction_text': """You are an aspect-based sentiment analyzer in Vietnamese. 
        # Given a text of user review, your task is to define all sentiment elements with their corresponding aspect categories and sentiment polarity.  
        # - The "aspect category" refers to the category that aspect belongs to.
        # - The "sentiment polarity" refers to the degree of positivity, negativity, or neutrality expressed in the opinion towards a particular aspect or feature of a product or service, 
        # and the available polarities include: "positive", "negative" and "neutral". "neutral" means mildly positive or mildly negative. Objective sentiment polarity should be ignored.
        # Your output MUST be in the format of a list of tuples as follows:
        # "SENTIMENT ELEMENTS: [(<aspect category: str>, <sentiment polarity: str>), ...]"
        # where each element of the tuple is enclosed by quotes. Do not output any other words.""",

        'examples_key': '### Examples:',
        'examples_text': """***Text: Rộng rãi KS mới nhưng rất vắng. Các dịch vụ chất lượng chưa cao và thiếu. ***Sentiments: {HOTEL#DESIGN&FEATURES, positive}, {HOTEL#GENERAL, negative}
        ***Text: Địa điểm thuận tiện, trong vòng bán kính 1,5km nhiều quán ăn ngon ***Sentiments: {LOCATION#GENERAL, positive}""",
        
        'input_key': '### Text:', 
        'input_text': "Ga giường không sạch, nhân viên quên dọn phòng một ngày.",
        
        'response_key': "### Sentiments:",
        'end_key': None
    }

    an_example_absa = """### Instruction:
    You are an aspect-based sentiment analyzer in Vietnamese. Given a text of user review, your task is to define all sentiment elements with their corresponding aspect categories and sentiment polarity.  

    - The "aspect category" refers to the category that aspect belongs to.
    - The "sentiment polarity" refers to the degree of positivity, negativity, or neutrality expressed in the opinion towards a particular aspect or feature of a product or service, 
    and the available polarities include: "positive", "negative" and "neutral". "neutral" means mildly positive or mildly negative. Objective sentiment polarity should be ignored.
    
    Your output MUST be in the format of a list of tuples as follows:
    "SENTIMENT ELEMENTS: [(<aspect category: str>, <sentiment polarity: str>), ...]"
    where each element of the tuple is enclosed by quotes. Do not output any other words.
    
    ### Categories
    ['FACILITIES#CLEANLINESS', 'FACILITIES#COMFORT', 'FACILITIES#DESIGN&FEATURES', 'FACILITIES#GENERAL', 'FACILITIES#MISCELLANEOUS', 'FACILITIES#PRICES', 'FACILITIES#QUALITY', 'FOOD&DRINKS#MISCELLANEOUS', 'FOOD&DRINKS#PRICES', 'FOOD&DRINKS#QUALITY', 'FOOD&DRINKS#STYLE&OPTIONS', 'HOTEL#CLEANLINESS', 'HOTEL#COMFORT', 'HOTEL#DESIGN&FEATURES', 'HOTEL#GENERAL', 'HOTEL#MISCELLANEOUS', 'HOTEL#PRICES', 'HOTEL#QUALITY', 'LOCATION#GENERAL', 'ROOMS#CLEANLINESS', 'ROOMS#COMFORT', 'ROOMS#DESIGN&FEATURES', 'ROOMS#GENERAL', 'ROOMS#MISCELLANEOUS', 'ROOMS#PRICES', 'ROOMS#QUALITY', 'ROOM_AMENITIES#CLEANLINESS', 'ROOM_AMENITIES#COMFORT', 'ROOM_AMENITIES#DESIGN&FEATURES', 'ROOM_AMENITIES#GENERAL', 'ROOM_AMENITIES#MISCELLANEOUS', 'ROOM_AMENITIES#PRICES', 'ROOM_AMENITIES#QUALITY', 'SERVICE#GENERAL']
    
    ### Text:
    Ga giường không sạch, nhân viên quên dọn phòng một ngày.
    
    ### Sentiments:
    {ROOM_AMENITIES#CLEANLINESS, negative}, {SERVICE#GENERAL, negative}
    """


    fewshot_prompt_kwargs = {
        'context_key': "### Context",
        'context_text': "CREATE TABLE city (Official_Name VARCHAR, Status VARCHAR, Population VARCHAR)",

        'examples_key': '### Examples',
        'examples_text': '',
        
        'instruction_key': "### Instruction:",
        'instruction_text': """You are an expert in data analysis using SQL. Given a question and context, your task is to write SQL query to find answer for the question .""",
        
        'input_key': '### Question:', 
        'input_text': "List the official name and status of the city with the largest population.",
        
        'response_key': "### SQL:",
        'end_key': None
    }


    generate_kwargs = {
        'max_new_tokens': 512,
        'skip_special_tokens': True
    }

    an_example_text2sql = """### Instruction:
    You are an expert in data analysis using SQL. Given a question and context, your task is to write SQL query to find answer for the question .
    
    ### Context
    CREATE TABLE city (Official_Name VARCHAR, Status VARCHAR, Population VARCHAR)
    
    ### Question:
    List all the cities in a decreasing order of each city's stations' highest latitude.
    
    ### SQL:
    SELECT Official_Name, Status FROM city ORDER BY Population DESC LIMIT 1
    """

    kwargs = {**model_kwargs, **absa_prompt_kwargs, **generate_kwargs}

    device = get_device()
    
    results = generate_response(use_cpu=False, return_full_text=True, do_postprocess=False, **kwargs)
    for result in results:
        print(result)