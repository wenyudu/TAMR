# Set up logging
import sys
sys.path.append('.')
import logging
from sqlalchemy import true
from flagai.trainer import Trainer
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)
import torch
import os
import json
from pathlib import Path
from contextlib import nullcontext
from dataclasses import asdict, fields
from transformers.hf_argparser import HfArgumentParser
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.models.auto import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.trainer_utils import get_last_checkpoint, set_seed
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from tokenizers import AddedToken
from seq2seq.utils.args import ModelArguments
from seq2seq.utils.picard_model_wrapper import PicardArguments, PicardLauncher, with_picard
from seq2seq.utils.dataset import DataTrainingArguments, DataArguments
from seq2seq.utils.dataset_loader import load_dataset
from seq2seq.utils.spider import SpiderTrainer
from seq2seq.utils.cosql import CoSQLTrainer

class FlagAISpiderTrainer(Trainer):
    def __init__(self, **kwargs):
        super(FlagAISpiderTrainer,self).__init__(**kwargs)
    def forward_step(self, data, model, mems):
        model_output = model(**data)
        
        return {"loss":model_output.loss,
                "logits": model_output.logits,
                "hidden_states": model_output.logits,
        }

trainer = FlagAISpiderTrainer(
    env_type='deepspeed',
    pytorch_device='cuda:1',  # "cuda:0" etc.
    experiment_name="t5-base",  # "The experiment name for summary and checkpoint"
    batch_size=24,  # 'Data Loader batch size'
    gradient_accumulation_steps=10000,  # 'Data Loader batch size'
    weight_decay=0.0,  # 'weight decay coefficient for L2 regularization'
    lr=1e-5,
    epochs=1,  # 'Number of finetunning epochs. Zero results in evaluation only.'
    save_interval=1000,  # 'number of epochs between saves')
    eval_interval=1e5, # do not do evaluate in trainer.train
    log_interval=10000,
    seed=1234,  # 'random seed'
    fp16=True,
    clip_grad=1.0,
    checkpoint_activations=False,

    # model checkpointing
    save_dir='checkpoints',  # 'Output directory to save checkpoints to.')
    save_optim=False,  # save current optimizer.')
    save_rng=False,  # save current rng state.')
    load_dir='checkpoints/99',  # Path to a directory containing a model checkpoint.')
    load_optim=False,  # not load optimizer when loading checkpoint.')
    # ' not load rng state when loading checkpoint.')):
    load_rng=False,
    tensorboard_dir="tensorboard_summary",
    master_ip='127.0.0.1',
    master_port=17755,
    num_nodes=1,
    num_gpus=1,
    hostfile='./hostfile',
    model_parallel_size=1,
    deepspeed_config='./deepspeed.json',
    training_script=__file__
)

# See all possible arguments by passing the --help flag to this script.
parser = HfArgumentParser(
    (PicardArguments, ModelArguments, DataArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
)
picard_args: PicardArguments
model_args: ModelArguments
data_args: DataArguments
data_training_args: DataTrainingArguments
training_args: Seq2SeqTrainingArguments
picard_args, model_args, data_args, data_training_args, training_args = parser.parse_json_file(
        json_file=os.path.abspath('configs/train.json')
    ) 
# if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
#     # If we pass only one argument to the script and it's the path to a json file,
#     # let's parse it to get our arguments.
#     picard_args, model_args, data_args, data_training_args, training_args = parser.parse_json_file(
#         json_file=os.path.abspath(sys.argv[1])
#     )
# elif len(sys.argv) == 3 and sys.argv[1].startswith("--local_rank") and sys.argv[2].endswith(".json"):
#     data = json.loads(Path(os.path.abspath(sys.argv[2])).read_text())
#     data.update({"local_rank": int(sys.argv[1].split("=")[1])})
#     picard_args, model_args, data_args, data_training_args, training_args = parser.parse_dict(args=data)
# else:
#     picard_args, model_args, data_args, data_training_args, training_args = parser.parse_args_into_dataclasses()

# If model_name_or_path includes ??? instead of the number of steps, 
# we load the latest checkpoint.
if 'checkpoint-???' in model_args.model_name_or_path:
    model_args.model_name_or_path = get_last_checkpoint(
        os.path.dirname(model_args.model_name_or_path))
    logger.info(f"Resolve model_name_or_path to {model_args.model_name_or_path}")

combined_args_dict = {
    **asdict(picard_args),
    **asdict(model_args),
    **asdict(data_args),
    **asdict(data_training_args),
    **training_args.to_sanitized_dict(),
}
combined_args_dict.pop("local_rank", None)

# if "wandb" in training_args.report_to and training_args.local_rank <= 0:
#     import wandb

#     init_args = {}
#     if "MLFLOW_EXPERIMENT_ID" in os.environ:
#         init_args["group"] = os.environ["MLFLOW_EXPERIMENT_ID"]
#     wandb.init(
#         project=os.getenv("WANDB_PROJECT", "text-to-sql"),
#         name=training_args.run_name,
#         **init_args,
#     )
#     wandb.config.update(combined_args_dict, allow_val_change=True)

# if not training_args.do_train and not training_args.do_eval and not training_args.do_predict:
#     logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
#     return

# Detect last checkpoint
last_checkpoint = None
if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )
    elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )

os.makedirs(training_args.output_dir, exist_ok=True)

if training_args.local_rank <= 0:
    with open(f"{training_args.output_dir}/combined_args.json", "w") as f:
        json.dump(combined_args_dict, f, indent=4)

# Initialize random number generators
set_seed(training_args.seed)

# Initialize config
config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
    max_length=data_training_args.max_target_length,
    num_beams=data_training_args.num_beams,
    num_beam_groups=data_training_args.num_beam_groups,
    diversity_penalty=data_training_args.diversity_penalty,
    gradient_checkpointing=training_args.gradient_checkpointing,
    use_cache=not training_args.gradient_checkpointing,
)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    use_fast=model_args.use_fast_tokenizer,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)
assert isinstance(tokenizer, PreTrainedTokenizerFast), "Only fast tokenizers are currently supported"
if isinstance(tokenizer, T5TokenizerFast):
    # In T5 `<` is OOV, see https://github.com/google-research/language/blob/master/language/nqg/tasks/spider/restore_oov.py
    tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])

# Load dataset
metric, dataset_splits = load_dataset(
    data_args=data_args,
    model_args=model_args,
    data_training_args=data_training_args,
    training_args=training_args,
    tokenizer=tokenizer,
)

# Initialize Picard if necessary
#with PicardLauncher() if picard_args.launch_picard and training_args.local_rank <= 0 else nullcontext(None):
    # Get Picard model class wrapper
if picard_args.use_picard:
    model_cls_wrapper = lambda model_cls: with_picard(
        model_cls=model_cls, picard_args=picard_args, tokenizer=tokenizer, schemas=dataset_splits.schemas
    )
else:
    model_cls_wrapper = lambda model_cls: model_cls

# Initialize model
model = model_cls_wrapper(AutoModelForSeq2SeqLM).from_pretrained(
    model_args.model_name_or_path,
    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    config=config,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)
if isinstance(model, T5ForConditionalGeneration):
    model.resize_token_embeddings(len(tokenizer))

if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
    logger.warning(
        "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
        f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
    )

# Initialize Trainer
trainer_kwargs = {
    "model": model,
    "args": training_args,
    "metric": metric,
    "train_dataset": dataset_splits.train_split.dataset if training_args.do_train else None,
    "eval_dataset": dataset_splits.eval_split.dataset if training_args.do_eval else None,
    "eval_examples": dataset_splits.eval_split.examples if training_args.do_eval else None,
    "tokenizer": tokenizer,
    "data_collator": DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=(-100 if data_training_args.ignore_pad_token_for_loss else tokenizer.pad_token_id),
        pad_to_multiple_of=8 if training_args.fp16 else None,
    ),
    "ignore_pad_token_for_loss": data_training_args.ignore_pad_token_for_loss,
    "target_with_db_id": data_training_args.target_with_db_id,
}

"""
overload the forward_step and evaluate function for NL2SQL
"""

# #using spidertrainer as it is
eval_dataset = trainer_kwargs['eval_dataset']
eval_examples = trainer_kwargs['eval_examples']
def _post_process_function(examples, features, tokenizer):
    import numpy as np
    inputs = tokenizer.batch_decode([f["input_ids"] for f in features], skip_special_tokens=True)
    label_ids = [f["labels"] for f in features]
    # Replace -100 in the labels as we can't decode them.
    _label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
    decoded_label_ids = tokenizer.batch_decode(_label_ids, skip_special_tokens=True)
    metas = [
        {
            "query": x["query"],
            "question": x["question"],
            "context": context,
            "label": label,
            "db_id": x["db_id"],
            "db_path": x["db_path"],
            "db_table_names": x["db_table_names"],
            "db_column_names": x["db_column_names"],
            "db_foreign_keys": x["db_foreign_keys"],
        }
        for x, context, label in zip(examples, inputs, decoded_label_ids)
    ]
    return label_ids, metas
eval_label_ids, eval_metas = _post_process_function(eval_examples, eval_dataset, tokenizer)
new_eval_data = []
for d,m in zip(trainer_kwargs['eval_dataset'],eval_metas):
    d['meta']=m
    new_eval_data.append(d)
class Eval_with_Meta(torch.utils.data.Dataset):
    def __init__(self, datas):
        self.datas= datas
    def __len__(self):
        return len(self.datas)
    def __getitem__(self, idx):
        data = self.datas[idx]
        return data
eval_with_meta = Eval_with_Meta(new_eval_data)
def metric_wrapper(prediction, labels, metas):
    
    eval_predictions = tokenizer.batch_decode(eval_predictions, skip_special_tokens=True)
    predictions = [pred.split("|", 1)[-1].strip() for pred in predictions]
    return metric.compute(predictions=predictions, references=metas)
    
# Training
if training_args.do_train:
    logger.info("*** Train ***")
    trainer.train(
        model = model,
        train_dataset=  trainer_kwargs['train_dataset'],
        valid_dataset=  eval_with_meta,
        collate_fn=trainer_kwargs['data_collator'],
        metric_methods= [['nl2sql',metric_wrapper]]
    )
    
# Evaluation
if training_args.do_eval:
    logger.info("*** Evaluate ***")
    
        
        

        # metrics = trainer.evaluate(
        #     max_length=data_training_args.val_max_target_length,
        #     max_time=data_training_args.val_max_time,
        #     num_beams=data_training_args.num_beams,
        #     metric_key_prefix="eval",
        # )
        # max_val_samples = (
        #     data_training_args.max_val_samples
        #     if data_training_args.max_val_samples is not None
        #     else len(dataset_splits.eval_split.dataset)
        # )
        # metrics["eval_samples"] = min(max_val_samples, len(dataset_splits.eval_split.dataset))

        # trainer.log_metrics("eval", metrics)
        # trainer.save_metrics("eval", metrics)

    # # Testing
    # if training_args.do_predict:
    #     logger.info("*** Predict ***")
    #     for section, test_split in dataset_splits.test_splits.items():
    #         results = trainer.predict(
    #             test_split.dataset, 
    #             test_split.examples,
    #             max_length=data_training_args.val_max_target_length,
    #             max_time=data_training_args.val_max_time,
    #             num_beams=data_training_args.num_beams,
    #             metric_key_prefix=section)
    #         metrics = results.metrics

    #         metrics[f"{section}_samples"] = len(test_split.dataset)

    #         trainer.log_metrics(section, metrics)
    #         trainer.save_metrics(section, metrics)

