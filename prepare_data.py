import os
import shutil
import joblib
import argparse
import yaml
from typing import Tuple
import logging

from functools import partial

from datasets import load_dataset

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
                          input_col: str="question",
                          output_col: str="answer",
                          context_col: str="context",
                          
                          instruction_key: str="### Instruction:",
                          instruction_text: str="You are a knowledgeable assistant for the company CMC Global. Your task is to providing accurate and helpful answers to the user's questions about the company.",

                          context_key: str="### Context:",
                          input_key: str = "### Question:",
                          response_key: str = "### Answer:",
                          end_key = None,
                          do_tokenize = False, 
                          max_length = None, 
):
    instruction = f'{instruction_key}\n{instruction_text}'
    
    context = f'{context_key}\n{example[context_col]}' if context_col else None


    input = f'{input_key}\n{example[input_col]}'

    response = f'{response_key}\n{example[output_col]}'
    
    if not end_key:
        end_key = tokenizer.eos_token
    
    end = f'{end_key}'
    
    if not use_model_chat_template:  # Not using default model chat template
        parts = [part for part in [instruction, context, input, response] if part]
        formatted_prompt = "\n\n".join(parts)
    else:   # Using defaut model chat template
        if has_system_role_support(tokenizer):

            if context_col:
                input = f'{context}\n{input}'
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": input},
                {"role": "assistant", "content": response},   
            ]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        else:
            if context_col:
                input = f'{context}\n{input}'
            messages = [
                {"role": "user", "content": instruction + '\n' + input},
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

def prepare_data(exp_args, data_args, model_args):

    if data_args.dataset.is_dataset_dict:
        dataset_dict = load_dataset(data_args.dataset.data_path,
                                    trust_remote_code=True)
    
    else:    
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
          input_col = data_args.dataset.input_col,
          output_col = data_args.dataset.output_col,
          context_col = data_args.dataset.context_col,
          
          instruction_key = data_args.prompt.instruction_key, # "### Instruction:",
          instruction_text = data_args.prompt.instruction_text, # "You are a knowledgeable assistant for the company CMC Global. Your task is to providing accurate and helpful answers to the user's questions about the company.",
          
          context_key =  data_args.prompt.context_key,
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


def setup_environment() -> None:
    from dotenv import load_dotenv
    # print("SETTING UP ENVIRONMENT...")
    _ = load_dotenv()

def main():

    from hydra import initialize, compose
    from hydra.utils import instantiate
    
    from omegaconf import OmegaConf
    
    from src.utils.utils import load_args

    from src.utils.log_utils import init_logging, setup_logger
    from src.utils.exp_utils import create_exp_dir

    # Setup logging
    # init_logging()
    logger = setup_logger("ft_llm")

    # Setup environment
    logger.info("SETTING UP ENVIRONMENT...")
    setup_environment()


    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process experiment configurations.')
    parser.add_argument(
        '--config_path',
        type=str,
        required=True,
        help='Path to the configuration file for the experiment.'
    )

    # args = parser.parse_args()
    args, override_args = parser.parse_known_args()  # Capture any extra positional arguments (overrides)

    logger.info("LOADING CONFIGS...")

    # Normalize the provided config path
    config_path = os.path.normpath(args.config_path)  # Normalize path for the current OS

    # Check if the configuration file exists
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    # Extract directory and filename from the provided config path
    config_dir = os.path.dirname(config_path)
    config_fn = os.path.splitext(os.path.basename(config_path))[0]

    # print(config_dir)
    # print(config_fn)

    try:
        with initialize(version_base=None, config_path=config_dir):
            # Compose the configuration with optional overrides
            cfg = compose(config_name=config_fn, overrides=override_args)
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration from {config_path}: {e}")

    # Print the configuration for debugging
    # print("type(cfg):", type(cfg))
    print(OmegaConf.to_yaml(cfg))

    assert os.path.basename(config_path).replace('.yaml', '') == cfg.exp_manager.exp_name
    
    logger.info("CREATING EXP DIR...")
    
    exp_name = cfg.exp_manager.exp_name
    # create_exp_dir(os.path.basename(config_path).replace('.yaml', ''))
    exp_dir, configs_dir, data_dir, checkpoints_dir, results_dir  = create_exp_dir(exp_name)

    import shutil
    shutil.copy(config_path, configs_dir)

    exp_args = cfg.exp_manager
    train_args = cfg.train
    data_args = cfg.prepare_data
    model_args = cfg.prepare_model


    if data_args.dataset.is_prepared:
        # Get the path to the processed data
        processed_data_path = os.path.normpath(data_args.dataset.prepared_data_path)
        
        # Check if the processed data exists
        if not os.path.isfile(processed_data_path):
            raise FileNotFoundError(f"Processed data not found at: {processed_data_path}")
        
        # Load the dataset
        logger.info("LOADING PROCESSED DATASET...")
        dataset = joblib.load(processed_data_path)
    else:
        # Prepare dataset
        logger.info("PREPARING DATASET...")
        from prepare_data import prepare_data
        dataset, processed_data_path = prepare_data(exp_args, data_args, model_args)

    # print(dataset)
    
    # Show dataset examples
    show_dataset_examples(dataset)


if __name__ == '__main__':
    main()
