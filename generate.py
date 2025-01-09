from src.utils.model_utils import get_model_tokenizer
from typing import List, Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Pipeline,
    PreTrainedModel,
    PreTrainedTokenizer,
)

example_question = """### Instruction:
You are a knowledgeable assistant for the company CMC Global. Your task is to providing accurate and helpful answers to the user's questions about the company.

### Question:
CMC Global cung cấp giải pháp gì cho khách hàng?

### Answer:
CMC Global cung cấp giải pháp IT một cửa, bao gồm IT Outsourcing, Chuyển đổi số và các giải pháp CMC với nhiều mô hình dịch vụ khác nhau."," :You are a customer AI who a company ""... Your task is to provide information and helpful information to the user's questions about the company.### Question:
"""


def load_model_for_generate(
    pretrained_model_name_or_path: str,
) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path, 
        # device_map="auto", 
        # torch_dtype=torch.bfloat16, 
        # trust_remote_code=True
    )

    return model

def load_tokenizer_for_generate(
    tokenizer_name_or_path: str
) -> PreTrainedTokenizer:
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    return tokenizer

def load_model_tokenizer_for_generate(
    pretrained_model_name_or_path: str,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Loads the model and tokenizer so that it can be used for generating responses.

    Args:
        pretrained_model_name_or_path (str): name or path for model

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, 
        # padding_side="left"
    )
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path, 
        # device_map="auto", 
        # torch_dtype=torch.bfloat16, 
        # trust_remote_code=True
    )
    return model, tokenizer


def prepare_prompt(question, **prompt_kwargs):
    from src.consts import GENERATION_PROMPT_FORMAT
    prompt = GENERATION_PROMPT_FORMAT.format(
        instruction_key=prompt_kwargs['instruction_key'],
        instruction_text=prompt_kwargs['instruction_text'],
        input_key=prompt_kwargs['input_key'],
        question=question,
        response_key=prompt_kwargs['response_key'],
    )

    return prompt

# def prepare_prompt(sample):
#     INTRO_BLURB = "You are a knowledgeable assistant for the company CMC Global. Your task is to providing accurate and helpful answers to the user's questions about the company."
#     INSTRUCTION_KEY = "### Question:"
#     INPUT_KEY = "### Context:"
#     RESPONSE_KEY = "### Answer:"
#     END_KEY = "### End"
    
#     blurb = f"{INTRO_BLURB}"
#     instruction = f"{INSTRUCTION_KEY}\n{sample['question']}"
#     try:
#         input_context = f"{INPUT_KEY}\n{sample['context']}" if sample["context"] else None
#     except:
#         input_context = None
#     response = f"{RESPONSE_KEY}\n{sample['answer']}"
#     end = f"{END_KEY}"
    
#     # parts = [part for part in [blurb, instruction, input_context, response, end] if part]
#     parts = [part for part in [blurb, instruction, input_context, response] if part]

#     formatted_prompt = "\n\n".join(parts)

#     sample["text"] = formatted_prompt

#     return sample


def postprocess(
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
    pattern = r"{}\s*(.+?){}".format(kwargs['response_key'], kwargs['end_key'])
    m = re.search(pattern, fully_decoded, flags=re.DOTALL)
    if m:
        decoded = m.group(1).strip()
    else:
        # The model might not generate the "### End" sequence before reaching the max tokens.  In this case,
        # return everything after "### Response:".
        pattern = r"{}\s*(.+)".format(kwargs['response_key'])
        m = re.search(pattern, fully_decoded, flags=re.DOTALL)
        if m:
            decoded = m.group(1).strip()
        else:
            # logger.warn(f"Failed to find response in:\n{fully_decoded}")
            print(f"Failed to find response in:\n{fully_decoded}")

    # if return_full_text:
    #     decoded = f"{}\n{decoded}"

    return decoded


def generate_response(question, tokenizer, model, **generate_kwargs):
    # sample = {
    #     "question": question,
    #     "answer": ""
    # }

    prompt = prepare_prompt(question, **generate_kwargs)
    # input = sample['text']
    input = tokenizer(prompt, return_tensors='pt', padding=False, truncation=True).to('cpu')
    output_ids = model.generate(**input, max_new_tokens=100, pad_token_id=tokenizer.pad_token_id)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    answer = postprocess(answer, **generate_kwargs)
    return answer

if __name__ == "__main__":
    tokenizer_name_or_path = 'exps/it__qwen2.5-0.5b-instruct__cmc_global_qa_480/results/tokenizer'
    tokenizer = load_tokenizer_for_generate(tokenizer_name_or_path)
    # print(tokenizer)

    pretrained_model_name_or_path = 'exps/it__qwen2.5-0.5b-instruct__cmc_global_qa_480/results/finetuned_model'
    model = load_model_for_generate(pretrained_model_name_or_path)
    # print(model)

    import argparse
    parser = argparse.ArgumentParser(description='Generate an answer for a given question.')
    parser.add_argument(
        '--question',
        type=str,
        required=True,
        help='The question to generate an answer for.'
    )
    args = parser.parse_args()

    generate_kwargs = {
        'instruction_key': '### Instruction:',
        'instruction_text': """You are a knowledgeable assistant for the company CMC Global. Your task is to providing accurate and helpful answers to the user's questions about the company.""",
        'input_key': '### Question:', 
        'response_key': '### Answer:',
        'end_key': '<|im_end|>'
    }
    result = generate_response(args.question, tokenizer, model, **generate_kwargs)
    print("Answer:", result)
