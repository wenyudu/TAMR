# Set up logging
import sys
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
from flagai.logger import log_dist
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
import deepspeed
import torch.distributed as dist
from tqdm import tqdm

class FlagAISpiderTrainer(Trainer):
    def forward_step(self, data, model, mems):
        tmp_data = {x:data[x] for x in data.keys() if x!='uid'}
        model_output = model(**tmp_data)
        
        return {"loss":model_output.loss
        }
    def evaluate_and_print_results(
        self,
        prefix=None,
        forward_step_func=None,
        data_loader=None,
        model=None,
        verbose=False,
        ):
        """Helper function to evaluate and dump results on screen."""
        eval_dict = self.evaluate(forward_step_func=forward_step_func,
                                  data_loader=data_loader,
                                  model=model,
                                  verbose=verbose)
        string=""
        if eval_dict.get("loss", None) is not None:
            string = ' validation loss at {} | {:.4f}, '.format(
                prefix, eval_dict["loss"])

        if self.metric_methods is None:
            return eval_dict

        # for i in range(len(self.metric_methods)):
            #name = self.metric_methods[i][0]
        for name in eval_dict.keys():
            if name!='loss':
                string += ", {} {:.4f}".format(name, eval_dict[name])
        # string = ' validation loss at {} | {:.4f},  Acc {:.2f}'.format(
        #     prefix, eval_dict["loss"], eval_dict["metrics"])
        length = len(string) + 1
        log_dist('-' * length, [0])
        log_dist(string, [0])
        log_dist('-' * length, [0])
        return eval_dict

    def evaluate(self,
                 data_loader=None,
                 model=None,
                 forward_step_func=None,
                 verbose=False):
        """Evaluation."""
        # Turn on evaluation mode which disables dropout.
        tmp_model = model
        while hasattr(tmp_model,'module'):
            tmp_model= tmp_model.module
        tmp_model.eval()

        # Turn off checkpoint_activations
        # tmp_checkpoint_activations = None
        # if self.env_type == 'pytorch' and self.fp16 is False:
        #     tmp_checkpoint_activations = model.config['checkpoint_activations']
        #     model.config['checkpoint_activations'] = False
        # elif self.fp16 is False:
        #     tmp_checkpoint_activations = model.module.config['checkpoint_activations']
        #     model.module.config['checkpoint_activations'] = False
        # else:
        #     tmp_checkpoint_activations = model.module.module.config['checkpoint_activations']
        #     model.module.module.config['checkpoint_activations'] = False
        
        mems = None
    
        from contextlib import nullcontext
        # with torch.no_grad():
        metric_dct = {}
        with nullcontext():
            assert data_loader is not None, "val loader is not None."
            all_logits = []
            all_labels = []
            all_ids = []
            all_losses = []
            all_generated_tokens = []
            for data_iterator in tqdm(data_loader):
                # Forward evaluation.

                meta = data_iterator.get('meta', None)
                
                if 'deepspeed' in self.env_type or 'DDP' in self.env_type:
                    data_iterator = {
                        x: data_iterator[x].to(
                            torch.device('cuda', self.local_rank))
                        for x in data_iterator
                        if x not in [ 'meta', 'mode']
                    }
                elif torch.cuda.is_available():

                    data_iterator = {
                        x:
                            data_iterator[x].to(torch.device(self.pytorch_device))
                        for x in data_iterator
                        if x not in [ 'meta', 'mode']
                    }
                
                with torch.no_grad():
                    step_output = forward_step_func(data_iterator,
                                                    model,
                                                    mems)
                '''when contiguous memory optimizations are enabled, the buffers
                allocated by the optimizations are deallocated during backward pass
                in the absence of backward pass the buffers should be reset after each
                forward pass'''
                if 'deepspeed' in self.env_type and self.deepspeed_activation_checkpointing:
                    deepspeed.checkpointing.reset()
  
                lm_loss = step_output['loss']
                labels = data_iterator['labels']
                ids = data_iterator['uid'].view(-1,1)
                
                if labels.shape[-1] < 512:
                    labels = self._pad_tensors_to_max_len(labels, 512)
                if ids.shape[0]<self.batch_size:
                    tmp_ids = torch.zeros((self.batch_size,1), dtype=torch.float32, device=ids.device)
                    tmp_ids[: ids.shape[0],:] = ids 
                    ids = tmp_ids

                # all_logits.append(logits)
                all_labels.append(labels)
                all_losses.append(lm_loss.view(1))
                all_ids.append(ids)

                gen_kwargs = {
                    "max_length": 512,
                    "num_beams": 1,
                    "synced_gpus": False,
                }
                if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
                    generation_inputs = data_iterator[self.model.encoder.main_input_name]
                else:
                    generation_inputs = data_iterator[self.model.main_input_name]

                generated_tokens = self.model.generate(
                    generation_inputs,
                    **gen_kwargs,
                )

                if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
                    generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
                all_generated_tokens.append(generated_tokens)

                # print(generated_tokens.shape)

            all_generated_tokens = torch.cat(all_generated_tokens, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_losses = torch.cat(all_losses, dim=0)
            all_ids = torch.cat(all_ids, dim=0).float()
          

            if all_losses.device != torch.device('cpu'):
                all_losses = all_losses.mean().cpu().detach().item()
            if all_ids.device!=torch.device('cpu'):
                all_ids= all_ids.cpu().detach().long().view(-1).numpy().tolist()
            # remove the duplicated results
            tmp_meta = []
            ids_recorder = set()
            tmp_results = []
            idx = 0
            for i,x in enumerate(all_ids):
                x = int(x)
                if x not in ids_recorder:
                    tmp_meta.append(self.metas[x])
                    tmp_results.append(all_generated_tokens[idx])
                    ids_recorder.add(x)
                    idx +=1
                elif x >0:
                    idx+=1

            eval_method = self.metric_methods[0][1]
            rs = eval_method(tmp_results, labels, meta=tmp_meta)
            exact_match = rs['exact_match']*len(all_ids)
            exec = rs['exec']*len(all_ids)
            results = torch.FloatTensor([exact_match, exec, float(len(all_ids))]).to(torch.device('cuda', self.local_rank))
            dist.all_reduce(results)
            metric_dict['exact_match']= results[0].item()/results[2].item()
            metric_dict['exec']= results[1].item()/results[2].item()
            del all_generated_tokens
            del all_labels

        # Move model back to the train mode.
        tmp_model.train()
        metric_dct.update({"loss":all_losses})
        return metric_dct

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor


trainer = FlagAISpiderTrainer(
    env_type='deepspeed',
    pytorch_device='cuda:1',  # "cuda:0" etc.
    experiment_name="t5",  # "The experiment name for summary and checkpoint"
    batch_size=16,  # 'Data Loader batch size'
    gradient_accumulation_steps=1,  # 'Data Loader batch size'
    weight_decay=0.0,  # 'weight decay coefficient for L2 regularization'
    lr=1e-4,
    warm_up=0,
    epochs=8,  # 'Number of finetunning epochs. Zero results in evaluation only.'
    save_interval=900000000,  # 'number of epochs between saves')
    eval_interval=1, # do not do evaluate in trainer.train
    # eval_interval= 72, # do not do evaluate in trainer.train
    log_interval=100,
    seed=11,  # 'random seed'
    fp16=False,
    clip_grad=1.0,
    checkpoint_activations=False,

    # model checkpointing
    save_dir='checkpoints',  # 'Output directory to save checkpoints to.')
    save_optim=False,  # save current optimizer.')
    save_rng=False,  # save current rng state.')
    load_dir=None,  # Path to a directory containing a model checkpoint.')
    load_optim=False,  # not load optimizer when loading checkpoint.')
    save_best = None,
    # ' not load rng state when loading checkpoint.')):
    load_rng=False,
    tensorboard_dir="tensorboard_summary",
    master_ip='127.0.0.1',
    master_port=17755,
    num_nodes=1,
    num_gpus=4,
    hostfile='./hostfile_eval',
    model_parallel_size=1,
    deepspeed_config='./deepspeed_internal_eval.json',
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
        json_file=os.path.abspath('configs/train_eval.json')
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
    gradient_checkpointing=False,
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
trainer.metas = eval_metas
trainer.tokenizer = tokenizer
trainer.model = model

new_eval_data = []
for i,m in enumerate(trainer_kwargs['eval_dataset']):
    m['uid']=i
    new_eval_data.append(m)

class Eval_with_Meta(torch.utils.data.Dataset):
    def __init__(self, datas):
        self.datas= datas
    def __len__(self):
        return len(self.datas)
    def __getitem__(self, idx):
        data = self.datas[idx]
        return data
eval_with_meta = Eval_with_Meta(new_eval_data)

def metric_wrapper(predictions, labels, meta=None):
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    predictions = [pred.split("|", 1)[-1].strip() for pred in predictions]
    return metric.compute(predictions=predictions, references=meta)
    #return metric.compute(predictions=predictions[:1034], references=meta)

# Training
if training_args.do_train:
    logger.info("*** Train ***")
    from flagai.optimizers import get_optimizer, get_optimizer_param_groups
    param_groups = get_optimizer_param_groups(model)

    if hasattr(param_groups[0], 'params'):
        # for T5 Model
        param_groups = param_groups[0]['params']

    optimizer = get_optimizer(
            param_groups=param_groups,
            lr=trainer.lr,
            weight_decay=trainer.weight_decay,
            cpu_optimizer=False,
            cpu_torch_adam=False,
            fp16=trainer.fp16,
            optimizer='adafactor') 
    split_size = int(len(eval_with_meta.datas)/trainer.batch_size)
    if trainer.rank != trainer.world_size-1:
        eval_with_meta.datas = eval_with_meta.datas[trainer.rank*split_size:(trainer.rank+1)*split_size]
    else:
        eval_with_meta.datas = eval_with_meta.datas[trainer.rank*split_size:] 
    eval_dataloader = torch.utils.data.DataLoader(eval_with_meta,
                                    batch_size=trainer.batch_size,
                                    num_workers=4,
                                    drop_last=False,
                                    pin_memory=False,
                                    prefetch_factor=4,
                                    collate_fn=trainer_kwargs['data_collator'])
    trainer.train(
        model = model,
        optimizer=optimizer,
        train_dataset=  trainer_kwargs['train_dataset'],
        valid_dataset= eval_dataloader, 
        collate_fn=trainer_kwargs['data_collator'],
        metric_methods= [['nl2sql',metric_wrapper]]
    )
    
        
        

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

