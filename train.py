import os
import glob
import torch
import logging

from utils.utils import set_env, barrier
from utils.processor import Processor
from utils.edit_trainer import EditDiffusionLMTrainer
from models import AutoTokenizer, AutoModelForCausalLM

from dataclasses import field, dataclass
from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import HfArgumentParser, TrainingArguments

logger = logging.getLogger()

@dataclass
class DiffusionTrainingArguments(TrainingArguments):

    # data
    max_len: int = field(default=4096)
    pad_len: int = field(default=None)
    num_workers: int = field(default=64)

    train_file: list[str] = field(default=None)

    # model
    model_cfg: str = field(default=None)

    # noisy
    min_t: float = field(default=0.0)
    max_t: float = field(default=1.0)

    intermediate_min_t: float = field(default=0.0)
    intermediate_max_t: float = field(default=1.0)
    
    # intermediate data
    intermediate_ratio: float = field(default=0.0)
    
    max_diffusion_steps: int = field(default=32)

    mask_diffusion_pred_per_step: list[int] = field(default_factory=lambda: [2, 4, 8])

    copy_head: bool = field(default=False)
    
    resume: bool = field(default=False)

    def __post_init__(self):
        super().__post_init__()
        self.gradient_checkpointing_kwargs = {"use_reentrant": False}

def get_model(args):

    model = AutoModelForCausalLM.from_pretrained(
        args.model_cfg,
        dtype=torch.bfloat16, 
        _attn_implementation="flash_attention_2"
    )

    logger.info(model)

    tokenizer = AutoTokenizer.from_pretrained(args.model_cfg)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    processor = Processor(args.max_len, args.pad_len, tokenizer)

    return model, tokenizer, processor

def get_files(path):
    results = []
    for root, _, files in os.walk(path):
        for file in files:
            if '.jsonl' in file:
                results.append(os.path.join(root, file))
    return results

def tokenize_dataset(args, processor, paths):

    with args.main_process_first(desc="dataset map tokenization", local=False):

        train_sets = []

        for path in paths:

            if os.path.exists(os.path.join(path, 'dataset_info.json')):
                dataset = load_from_disk(path)
                logger.info('Load dataset from disk %s', path)
            else:
                if os.path.isdir(path):
                    src = get_files(path)
                else:
                    src = glob.glob(path)
                
                logger.info("Loading dataset from files: %s", src)
                dataset = load_dataset('json', data_files=src, split='train')
                dataset = dataset.map(
                    processor.process_tokenize,
                    batched=True,
                    batch_size=8192,
                    num_proc=args.num_workers,
                    remove_columns=list(dataset.features),
                    desc="Running tokenizer on dataset",
                )
            
            logger.info("Dataset size: %d", len(dataset))
            train_sets.append(dataset)
    
        train_sets = concatenate_datasets(train_sets)
    
    logger.info("Train dataset size: %d", len(train_sets))

    return train_sets

def train():
    parser = HfArgumentParser(DiffusionTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    set_env(args)

    model, tokenizer, processor = get_model(args)

    if args.copy_head:
        model.next_token_head.weight.data.copy_(model.lm_head.weight.data)

    train_set = tokenize_dataset(args, processor, args.train_file)
    trainer = EditDiffusionLMTrainer(
        args=args,
        model=model, 
        processing_class=tokenizer,
        train_dataset=train_set,
    )

    has_checkpoint = len(glob.glob(os.path.join(args.output_dir, "checkpoint-*"))) > 0

    trainer.train(resume_from_checkpoint=has_checkpoint)
    trainer.save_model(os.path.join(args.output_dir, "checkpoint-final"))

    barrier()

if __name__ == "__main__":

    try:
        train()
    except Exception as e:
        logging.exception(e)

        exit(-1)
