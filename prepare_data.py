# %%writefile prepare_data.py
import os
import shutil
import joblib
import argparse
import yaml
from typing import Tuple
import logging

from functools import partial

from src.utils.utils import load_args, update_yaml
from src.utils.model_utils import load_tokenizer


def load_template(template_file):
    """Load the template from a text file."""
    with open(template_file, 'r') as file:
        return file.read()


import pandas as pd
from datasets import Dataset, DatasetDict

def create_dataset_dict(data_path, 
                        do_split: bool=True, 
                        val_ratio: float=0.25,
                        test_ratio: float=0.2,
                        seed: int = 42):
    df = pd.read_csv(data_path)
    hf_dataset = Dataset.from_pandas(df)

  
    if do_split:
        # Splitting the dataset
        train_test_split = hf_dataset.train_test_split(test_size=test_ratio, seed=seed)
        train_val_split = train_test_split["train"].train_test_split(test_size=val_ratio, seed=seed)  # 0.25 x 0.8 = 0.2
    
        # Create DatasetDict
        dataset_dict = DatasetDict({
            "train": train_val_split["train"],
            "val": train_val_split["test"],
            "test": train_test_split["test"]
        })
        
    else:
        dataset_dict = DatasetDict({
            "train": hf_dataset
        })
    return dataset_dict

# def show_dataset_examples(dataset_dict):
#     """
#     Prints the length, columns, and an example from each split of a DatasetDict (train, val, test).

#     Parameters
#     ----------
#     dataset_dict : datasets.DatasetDict
#         A DatasetDict containing train, val, and test splits.

#     Returns
#     -------
#     None
#     """
#     for split_name, dataset in dataset_dict.items():
#         # Get the length and columns of the current split
#         dataset_length = len(dataset)
#         dataset_columns = dataset.column_names

#         print(f"\nSplit: {split_name}")
#         print(f"Number of Examples: {dataset_length}")
#         print(f"Columns: {dataset_columns}")

#         # Get the first example from the current split
#         example = dataset[0]

#         print(f"An example:")
#         for key, value in example.items():
#             print(f"  {key}: {value}")

#     print("\n" + "=" * 40 + "\n")

def show_dataset_examples(dataset_dict):
    """
    Prints the length, columns, shape of columns, and an example from each split of a DatasetDict (train, val, test).

    Parameters
    ----------
    dataset_dict : datasets.DatasetDict
        A DatasetDict containing train, val, and test splits.

    Returns
    -------
    None
    """
    for split_name, dataset in dataset_dict.items():
        # Get the length and columns of the current split
        dataset_length = len(dataset)
        dataset_columns = dataset.column_names

        print(f"\nSplit: {split_name}")
        print(f"Number of Examples: {dataset_length}")
        print(f"Columns: {dataset_columns}")

        # Calculate the shape of each column
        print("Shapes:")
        for column_name in dataset_columns:
            if column_name in dataset[0]:
                col_data = dataset[column_name]
                if isinstance(col_data[0], list):  # Multi-dimensional data (e.g., tokenized inputs)
                    print(f"  {column_name}: [{len(col_data)}, {len(col_data[0])}]")
                else:  # Single-dimensional data (e.g., strings)
                    print(f"  {column_name}: [{len(col_data)}]")

        # Get the first example from the current split
        example = dataset[0]

        print("An example:")
        for key, value in example.items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 40 + "\n")


def has_system_role_support(tokenizer):
    messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Who won the FIFA World Cup 2022?"
            },
            {
                "role": "assistant",
                "content": "Argentina won the FIFA World Cup 2022, defeating France in the final."
            }
    ]

    try: 
        tokenizer.apply_chat_template(messages, tokenize=False)
        return True
    except:
        return False



def create_prompt_formats(example,
                          tokenizer,
                          use_model_chat_template,
                          instruction_key: str = "### Instruction:",
                          instruction_text: str = "You are a knowledgeable assistant for the company CMC Global. Your task is to providing accurate and helpful answers to the user's questions about the company.",
                          input_key: str = "### Question:",
                          response_key: str = "### Answer:",
                          end_key = None,
                          do_tokenize = False, 
                          max_length = None, 
):
    instruction = f'{instruction_key}\n{instruction_text}'
    input = f'{input_key}\n{example["question"]}'

    response = f'{response_key}\n{example["answer"]}'
    
    if not end_key:
        end_key = tokenizer.eos_token
    
    end = f'{end_key}'
    
    if not use_model_chat_template:  # Not using default model chat template
        parts = [part for part in [instruction, input, response] if part]
        formatted_prompt = "\n\n".join(parts)
    else:   # Using defaut model chat template
        if has_system_role_support(tokenizer):
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": input},
                {"role": "assistant", "content": response},   
            ]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        else:
            messages = [
                {"role": "user", "content": instruction+ '\n' +input},
                {"role": "assistant", "content": response},
            ]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    if formatted_prompt.strip().endswith(end):
        example['text'] = formatted_prompt
        # print('endswith end')
    else:
        # print('does not ends with end')
        example['text'] = formatted_prompt + end

    if do_tokenize:
        tokenized_text = tokenizer(formatted_prompt, 
                                   truncation=True, 
                                   padding='max_length', 
                                   add_special_tokens=True, 
                                   max_length=max_length)
        
        example['input_ids'] = tokenized_text['input_ids']
        example['attention_mask'] = tokenized_text['attention_mask']

    return example
        
    
def save_dataset(dataset, save_path):
    joblib.dump(dataset, save_path)
    
def get_data_collator():
    pass
    
# def prepare_data(dataset_dict, template_file, keys):
#     """Prepare the dataset by formatting the data using the provided template."""
#     # Sample dataset (this should be loaded from your actual data source)
#     # dataset_dict = {
#     #     "train": [
#     #         {"question": "What is the capital of France?", "answer": "Paris", "question_type": "geography"},
#     #         {"question": "Who is the CEO of Tesla?", "answer": "Elon Musk", "question_type": "business"},
#     #     ]
#     # }

#     # Load the template from the file
#     template = load_template(template_file)

#     # Prepare the output list
#     output_texts = []

#     # Format the texts based on the template
#     for split_name in dataset_dict.keys():
#         for example in dataset_dict[split_name]:
            
#             formatted_text = template.format(
#                 **{key: example[key] for key in keys}
#             )

            
#             output_texts.append(formatted_text)

#     # Save the processed texts to a file
#     with open('processed_data.txt', 'w') as f:
#         f.write("\n\n".join(output_texts))

#     print("Data processing complete. Output saved to 'processed_data.txt'.")

def prepare_data(exp_args, data_args, model_args):

    dataset_dict = create_dataset_dict(data_args.dataset.data_path, 
                                       data_args.dataset.do_split, 
                                       data_args.dataset.val_ratio, 
                                       data_args.dataset.test_ratio, 
                                       exp_args.seed)
    
    tokenizer = load_tokenizer(data_args, model_args)

    _create_prompt_formats = partial(
        create_prompt_formats,
          tokenizer = tokenizer,
          use_model_chat_template = data_args.prompt.use_model_chat_template,
          instruction_key = data_args.prompt.instruction_key, # "### Instruction:",
          instruction_text = data_args.prompt.instruction_text, # "You are a knowledgeable assistant for the company CMC Global. Your task is to providing accurate and helpful answers to the user's questions about the company.",
          input_key = data_args.prompt.input_key, #"### Question:",
          response_key = data_args.prompt.response_key, #"### Answer:",
          end_key = data_args.prompt.end_key,
          do_tokenize = data_args.tokenizer.do_tokenize, 
          max_length = data_args.tokenizer.max_length
    )

    dataset = dataset_dict.map(
        _create_prompt_formats, 
         batched=False, 
         remove_columns=dataset_dict['train'].column_names
    )

    if data_args.dataset.do_save:
        save_path = data_args.dataset.prepared_data_path
        save_dataset(dataset, save_path)

    return dataset, save_path

if __name__ == "__main__":
    # Load parameters from params.yaml
    exp_args = load_args('configs/exp.yaml')

    data_cfg_path = exp_args.exp_manager.prepare_data_cfg_path
    data_args = load_args(data_cfg_path)

    train_cfg_path = exp_args.exp_manager.train_cfg_path
    train_args = load_args(train_cfg_path)

    model_args = train_args.model_args
    

    exps_dir = exp_args.exp_manager.exps_dir
    exp_name = exp_args.exp_manager.exp_name
    os.makedirs(exps_dir, exist_ok=True)

    exp_dir = os.path.join(exps_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    exp_config_dir = os.path.join(exp_dir, 'configs')
    os.makedirs(exp_config_dir, exist_ok=True)

    shutil.copy('configs/exp.yaml', exp_config_dir)
    shutil.copy('configs/train.yaml', exp_config_dir)
    shutil.copy(data_cfg_path, exp_config_dir)

    # shutil.copytree('./configs', exp_config_dir)

    data_dir = os.path.join(exp_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    dataset_dict = create_dataset_dict(data_args.dataset.data_path, 
                                       data_args.dataset.do_split, 
                                       data_args.dataset.val_ratio, 
                                       data_args.dataset.test_ratio, 
                                       exp_args.exp_manager.seed)

    # show_dataset_examples(dataset_dict)

    # from utils import load_tokenizer
    # tokenizer = load_tokenizer(data_args.tokenizer.model_name_or_path)
    logging.info("Loading tokenizer...")
    tokenizer = load_tokenizer(data_args, model_args)


    # from functools import partial

    _create_prompt_formats = partial(
        create_prompt_formats,
          tokenizer = tokenizer,
          use_model_chat_template = data_args.prompt.use_model_chat_template,
          instruction_key = "### Instruction:",
          instruction_text = "You are a knowledgeable assistant for the company CMC Global. Your task is to providing accurate and helpful answers to the user's questions about the company.",
          input_key = "### Question:",
          response_key = "### Answer:",
          end_key = data_args.prompt.end_key,
          do_tokenize = data_args.tokenizer.do_tokenize, 
          max_length = data_args.tokenizer.max_length
    )
    # if params.tokenizer.do_tokenize:
    dataset = dataset_dict.map(
        _create_prompt_formats, 
         batched=False, 
         remove_columns=dataset_dict['train'].column_names
    )
    output_path = data_args.dataset.output_path
    if not output_path:
        save_path =  os.path.join(data_dir, f'{exp_name}.pkl')
        update_yaml(os.path.join(exp_config_dir, os.path.basename(data_cfg_path)), 'dataset.output_path', save_path)
        
    else:
        save_path = output_path
         
    save_dataset(dataset, save_path)

    show_dataset_examples(dataset)

    # shutil.copy(data_cfg_path, exp_config_dir)
    
    # print(tokenized_dataset)
    # print(tokenized_dataset['train'][0])
    # show_dataset_examples(tokenized_dataset)
    
    # prepare_data(dataset_dict, custom_template_path, params.prompt.template_keys)
