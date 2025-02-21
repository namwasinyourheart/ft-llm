# %%writefile eval.py
import argparse
import os
import joblib
import re

from src.utils.exp_utils import setup_environment, create_exp_dir
from transformers import set_seed

import numpy as np
np.iinfo(np.int32)


from hydra import initialize, compose
from hydra.utils import instantiate

from omegaconf import OmegaConf


from src.utils.log_utils import setup_logger
from src.utils.exp_utils import create_exp_dir

from prepare_data import prepare_data, show_dataset_examples


def load_cfg(config_path, override_args=None, print_cfg=True):

    """
    Load a configuration file using Hydra and OmegaConf.
    
    Args:
        config_path (str): Path to the configuration file.
        override_args (list, optional): List of arguments to override configuration values.

    Returns:
        cfg: Loaded configuration object.
    """

    override_args = override_args or []
    config_path = os.path.normpath(config_path)
    
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    config_dir = os.path.dirname(config_path)
    config_fn = os.path.splitext(os.path.basename(config_path))[0]
    
    try:
        with initialize(version_base=None, config_path=config_dir):
            cfg = compose(config_name=config_fn, overrides=override_args)
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration from {config_path}: {e}")
    
    # assert os.path.basename(config_path).replace('.yaml', '') == cfg.exp_manager.exp_name, \
    # assert cfg.exp_manager.phase_name + '__' + 
    assert cfg.exp_manager.exp_name == os.path.basename(config_path).replace('.yaml', ''), \
    f"Config file name '{os.path.basename(config_path)}' does not match experiment name '{cfg.exp_manager.exp_name}' in the config."

    if print_cfg:
        print(OmegaConf.to_yaml(cfg))
    
    exp_args = cfg.exp_manager
    data_args = cfg.prepare_data
    model_args = cfg.prepare_model
    train_args = cfg.train
    eval_args = cfg.eval
    gen_args = cfg.generate

    return cfg, exp_args, data_args, model_args, train_args, eval_args, gen_args

def extract_answer(text, min_value=np.iinfo(np.int32).min, max_value=np.iinfo(np.int32).max):
    try:
        match = re.search(r"Answer:\s*(\d+)", text)
        if match:
            answer =  match.group(1)
            answer = answer.replace(',', '')

            if min_value <= int(float(answer)) <= max_value: 
                return answer
            
            else:
                return '-1'
            # answer = int(answer)    
        else:
            answer = "-1"
        return answer
    except Exception:
        return "-1"

def main():

    # Setup logging
    logger = setup_logger()

    # Setup environment
    logger.info("SETTING UP ENVIRONMENT...")
    setup_environment()


    # Parse arguments
    parser = argparse.ArgumentParser(description='Load experiment configurations.')
    parser.add_argument(
        '--config_path',
        type=str,
        required=True,
        help='Path to the configuration file for the experiment.'
    )

    args, override_args = parser.parse_known_args()

    # Load configuration
    logger.info("LOADING CONFIGURATIONS...")
    cfg, exp_args, data_args, model_args, train_args, eval_args, gen_args = load_cfg(config_path=args.config_path, override_args=override_args)
    
    # Create experiment directories
    # logger.info("CREATING DIRECTORIES...")""
    exp_name = cfg.exp_manager.exp_name
    (exp_dir, exp_data_dir, exp_checkpoints_dir, exp_results_dir) = create_exp_dir(exp_name)

    import shutil
    shutil.copy(args.config_path, exp_dir)

    # Set seed
    set_seed(exp_args.seed)


    #Load dataset
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
        dataset, processed_data_path = prepare_data(exp_args, data_args, model_args)
    
    # print(dataset)
        
    # Show dataset examples
    # show_dataset_examples(dataset)

    from torch.utils.data import DataLoader

    test_ds = dataset['test']
    test_ds = test_ds.with_format("torch")

    test_loader = DataLoader(test_ds, batch_size=eval_args.batch_size, shuffle=False)

    
    import torch
    
    from tqdm.auto import tqdm
    
    
    # from generate import load_model_for_generate, load_tokenizer_for_generate
    from transformers import PreTrainedTokenizer, AutoTokenizer, PreTrainedModel, AutoModelForCausalLM, BitsAndBytesConfig

    from src.utils.model_utils import set_torch_dtype_and_attn_implementation, get_quantization_config
    
    def load_tokenizer_for_generate(
        **model_args
    ) -> PreTrainedTokenizer:
        
        tokenizer = AutoTokenizer.from_pretrained(model_args['pretrained_tokenizer_name_or_path'])
        tokenizer.padding_side = 'left'
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    
        return tokenizer

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
            ).to_dict()
        else:
            quantization_config = None
    
        return quantization_config
    
    def load_model_for_generate(
        use_cpu: bool=False,
        **model_args,
    ) -> PreTrainedModel:
        
        torch_dtype, attn_implementation = set_torch_dtype_and_attn_implementation()
    
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_args['pretrained_model_name_or_path'],
            trust_remote_code=True,
            quantization_config=get_quantization_config(**model_args) if not use_cpu else None,
            device_map="cpu" if use_cpu else "auto",
            torch_dtype="auto",
            attn_implementation=attn_implementation,
            # low_cpu_mem_usage=True if not use_cpu else False
        )
    
        return model

    
    tokenizer = load_tokenizer_for_generate(**model_args)
    model = load_model_for_generate(**model_args)
    
    
    import evaluate
    accuracy_metric = evaluate.load("accuracy")
    model.eval()
    
    prediction_file = os.path.join(exp_results_dir, eval_args.prediction_file)
    with open(prediction_file, "w", encoding="utf-8") as f:
        
        with torch.no_grad():
          for step, batch in enumerate(tqdm(test_loader)):
              if step == 1:
                  break
              input_ids = batch['input_ids'].squeeze(1).to(model.device)
              attention_mask = batch['attention_mask'].squeeze(1).to(model.device)
                
              output_ids = model.generate(input_ids=input_ids, 
                                          attention_mask=attention_mask, 
                                          pad_token_id=tokenizer.eos_token_id,
                                         max_new_tokens=gen_args['max_new_tokens'])
              
              pred_answers = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
              pred_answers_e = [extract_answer(answer) for answer in pred_answers ]
              
              true_answers = batch['answer']
              true_answers = [answer.replace(',', '') for answer in true_answers]
              try:
                  accuracy_metric.add_batch(predictions=pred_answers_e, references=true_answers)
              except:
                  print(pred_answers)
                  print(true_answers)
                  continue
              
              
              indexs = batch['index']
              # Write prediction to file
              for index, raw_pred, proc_pred, true_ans in zip(indexs, pred_answers, pred_answers_e, true_answers):
                label = True if proc_pred == true_ans else False # 1 = Correct, 0 = Incorrect
                
                f.write(f'Index: {index}\n')
                f.write("-" * 12 + "\n")

                f.write(f"Prediction Text:\n")
                f.write(f"{raw_pred}\n")
                f.write("-" * 12 + "\n")

                f.write(f"Prediction Answer\n")
                f.write(f"{proc_pred}\n")
                f.write("-" * 12 + "\n")

                f.write(f"True Answer: {true_ans}\n")
                f.write("-" * 12 + "\n")

                f.write(f"Correct?: {label}\n")
                # f.write("\n\n")
                f.write("-" * 96 + "\n\n")

          

    # Compute final accuracy
    results = accuracy_metric.compute()
    print(f"Accuracy: {results['accuracy']}")

if __name__ == "__main__":
    main()
        