import os
import argparse
import time


import joblib
import wandb

from hydra import initialize, compose
from hydra.utils import instantiate

from omegaconf import OmegaConf


from src.utils.utils import load_args
from prepare_data import show_dataset_examples

# import logging
from dotenv import load_dotenv

from src.utils.model_utils import get_model_tokenizer, get_peft_config
from src.utils.log_utils import init_logging, setup_logger

from peft import (
    LoraConfig, 
    prepare_model_for_kbit_training,
    get_peft_model
)

from trl import (
    DataCollatorForCompletionOnlyLM, 
    SFTTrainer
)

from transformers import DataCollatorForLanguageModeling, set_seed

def to_linux_path(path):
    """Convert a given path to Linux-style with forward slashes."""
    return path.replace("\\", "/")

# def init_logging() -> None:
#     """Initialize logging."""
#     logging.basicConfig(
#         format="%(asctime)s %(levelname)s: %(message)s",
#         level=logging.INFO,
#         datefmt="%Y-%m-%d %H:%M:%S",
#     )

def setup_environment() -> None:
    # print("SETTING UP ENVIRONMENT...")
    _ = load_dotenv()

def main():

    # Setup logging
    # init_logging()
    logger = setup_logger("ft_llm")

    # Setup environment
    logger.info("SETTING UP ENVIRONMENT...")
    setup_environment()



    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process experiment configurations.')
    parser.add_argument(
        '--exp_name',
        type=str,
        help='Name of the experiment. This will determine the configuration file path.'
    )
    
    # args = parser.parse_args()
    args, override_args = parser.parse_known_args()  # Capture any extra positional arguments (overrides)

    logger.info("LOADING CONFIGS...")

    exp_dir = os.path.join('exps', args.exp_name)

    config_dir = os.path.join(exp_dir, 'configs')
    config_fn = 'exp'

    try:
        with initialize(version_base=None, config_path=config_dir):
            # Compose the configuration with optional overrides
            cfg = compose(config_name=config_fn, overrides=override_args)
            # cfg = OmegaConf.to_container(cfg, resolve=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration for {exp_dir}: {e}")
    
    # print("cfg:", cfg)

    print("type(cfg):", type(cfg))
    print(OmegaConf.to_yaml(cfg))
    
    
    # Construct the path to the experiment config file
    exp_cfg_path = os.path.join(args.exp_name, 'configs', 'exp.yaml')
    exp_cfg_path = os.path.join('exps', exp_cfg_path)
    exp_cfg_path = os.path.normpath(exp_cfg_path)  # Normalize path for the current OS
    
    # Check if the experiment config file exists
    if not os.path.isfile(exp_cfg_path):
        raise FileNotFoundError(f"Experiment configuration file not found at: {exp_cfg_path}")
    
    # Load experiment arguments
    # exp_args = load_args(exp_cfg_path, verbose=True)
    exp_args = cfg
    
    # # Extract paths from experiment arguments
    # data_cfg_path = os.path.join(exp_dir, exp_args.exp_manager.prepare_data_cfg_path)
    # data_cfg_path = os.path.normpath(data_cfg_path)

    # train_cfg_path = os.path.join(exp_dir, exp_args.exp_manager.train_cfg_path)
    # train_cfg_path = os.path.normpath(train_cfg_path)
    
    # # Load training and data arguments
    # train_args = load_args(to_linux_path(train_cfg_path), verbose=True)
    # data_args = load_args(to_linux_path(data_cfg_path),verbose=True)
    # model_args = train_args.model_args

    train_args = cfg.train
    data_args = cfg.prepare_data
    model_args = train_args.model_args

    
    # Get the path to the processed data
    processed_data_path = os.path.normpath(data_args.dataset.output_path)
    
    # Check if the processed data exists
    if not os.path.isfile(processed_data_path):
        raise FileNotFoundError(f"Processed data not found at: {processed_data_path}")
    
    # Load the dataset
    logger.info("LOADING PROCESSED DATASESS...")
    dataset = joblib.load(processed_data_path)

    # print(dataset)
    
    # Show dataset examples
    show_dataset_examples(dataset)

    # Set seed before initializing model.
    set_seed(exp_args.exp_manager.seed)

    # LOADING MODEL
    logger.info("LOADING MODEL AND TOKENIZER...")
    model, tokenizer = get_model_tokenizer(data_args, model_args, use_cpu=True)

    # print(model.config)
    # print(tokenizer)

    from src.metrics import calc_bleu, calc_rouge

    def compute_metrics(predictions):
        import numpy as np
        import nltk
        from nltk.translate.bleu_score import sentence_bleu
        # squad_labels = pred.label_ids
        # squad_preds = pred.predictions.argmax(-1)
    
        # # Calculate Exact Match (EM)
        # em = sum([1 if p == l else 0 for p, l in zip(squad_preds, squad_labels)]) / len(squad_labels)
    
        # # Calculate F1-score
        # f1 = f1_score(squad_labels, squad_preds, average='macro')
        
        # references = pred.label_ids
        # generated_texts = pred.predictions
    
        # print("references:", references)
        # print("generated_texts:", generated_texts)
    
        label_ids = predictions.label_ids
        label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        predictions =  predictions.predictions
        # print("predictions:", predictions)
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        logits = predictions.argmax(axis=-1)
        predictions = tokenizer.batch_decode(logits, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
        # print("predictions:", prediction_text)
        # print("labels:", labels)
        # Calculate bleu_scores using NLTK
        nltk_bleu_scores = []
        for reference, prediction in zip(labels, predictions):
            nltk_bleu_score = sentence_bleu([reference], prediction)
            nltk_bleu_scores.append(nltk_bleu_score)
        nltk_bleu = sum(nltk_bleu_scores) / len(nltk_bleu_scores)

        references = labels
        # Calculate bleu score using evaluate
        bleu_score = calc_bleu(predictions, references)

        # print("bleu_score:", bleu_score)
        
        rouge_score = calc_rouge(predictions, references)
        # print("rouge_score:", rouge_score)

        # results = {'nltk_bleu': nltk_bleu}.update(bleu_score.update(rouge_score))
        # results = {'nltk_bleu': nltk_bleu}
        results = bleu_score.copy()
        results.update(rouge_score)
        results.update({"nltk_bleu": nltk_bleu})

        # print("results:", results)
        
        return results

    # PREPARE MODEL
    # Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    peft_config = get_peft_config(model_args)
    print(peft_config)

    model = get_peft_model(model, peft_config)

    # Print information about the percentage of trainable parameters
    try:
        model.print_trainable_parameters()
    except:
        from src.utils.model_utils import print_trainable_parameters
        print_trainable_parameters(model)


    data_collator = DataCollatorForLanguageModeling(
        # response_template=RESPONSE_KEY,
        tokenizer=tokenizer, 
        mlm=False, 
        return_tensors="pt", 
        # pad_to_multiple_of=cfg.data.pad_to_multiple_of
    )

    # if exp_args.wandb.use_wandb:
    wandb.init(
        project=exp_args.wandb.project
    )

    current_date = time.strftime("%Y%m%d_%H%M%S")
    checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
    results_dir = os.path.join(exp_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    training_args = instantiate(train_args.train_args, 
                                output_dir=checkpoints_dir, 
                                report_to="none",
                                # run_name=wandb.run.name
                    )
    
    print(training_args)
    
    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Initialize our Trainer
    logger.info("Instantiating Trainer")
    # print("Instantiating Trainer")
    
    train_ds, val_ds, test_ds = dataset['train'], dataset['val'], dataset['test']
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds.select(range(5)),
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # print(training_args)

    model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs

    from src.callbacks import WandbPredictionProgressCallback, decode_predictions

    progress_callback = WandbPredictionProgressCallback(
        trainer=trainer,
        tokenizer=tokenizer,
        val_dataset=val_ds,
        num_samples=10,
        freq=1,
    )

    # Add the callback to the trainer
    trainer.add_callback(progress_callback)

    all_metrics = {"run_name": wandb.run.name}

    # Check for last checkpoint
    from src.utils.model_utils import get_checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    #  TRAINING
    if training_args.do_train:
        logger.info("TRAINING...")

        if training_args.resume_from_checkpoint:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        all_metrics.update(metrics)
        # print(metrics)

        # trainer.log_metrics("train", metrics)
        # trainer.save_metrics("train", metrics)
        # trainer.save_state()
        # all_metrics.update(metrics)

        # Save model
        logger.info("SAVING MODEL...")
        trainer.model.save_pretrained(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")

    import pandas as pd
    # PREDICTION
    if training_args.do_predict:
        logger.info("PREDICTING...")
        predictions = trainer.predict(test_ds.select(range(2)))
        # print("DECODING PREDICTION...")
        predictions = decode_predictions(tokenizer, predictions)
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(os.path.join(results_dir, 'predictions_df.csv'), index=False)
    
    import json
    
    if (training_args.do_train or training_args.do_eval or training_args.do_predict):
            with open(os.path.join(results_dir, "metrics.json"), "a") as fout:
                fout.write(json.dumps(all_metrics))

    logger.info("TRAINING COMPLETED.")
if __name__ == '__main__':
    main()
