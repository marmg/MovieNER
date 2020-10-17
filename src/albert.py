import argparse
import glob
import logging
import os
import json
import random
from types import SimpleNamespace
import pickle as pkl

import numpy as np
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch
# from seqeval.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm, trange

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from transformers import AlbertConfig, AlbertForTokenClassification, AlbertTokenizer, AlbertForQuestionAnswering
from transformers import squad_convert_examples_to_features
from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits)
from transformers.data.processors.squad import SquadResult, SquadV2Processor

from .utils_ner import convert_examples_to_features, read_examples_from_file, get_labels, parse_result
from .config import *
from .metrics import ner_f1_score, ner_precision_score, ner_recall_score
from .logging_handler import logger


def tokenize_sentence_wbl(sentence):
    words = sentence.split()
    words = [word.replace(".", "") if word.endswith(".") else word for word in words]

    return "\n".join(words)


class AlbertQA:

    def __init__(self):
        super().__init__()
        print("--- Albert for QA ---")

    def load(self, path):
        """ Method for pretrained model loading. """
        if not os.path.exists(path):
            raise NotADirectoryError(
                f"{os.path.abspath(path)} must be a directory containing the model files: config, tokenizer, weights.")

        files = os.listdir(path)
        if CONFIG_JSON_FILE not in files:
            raise FileNotFoundError(f"{CONFIG_JSON_FILE} must be in {path}.")
        if WEIGHTS_FILE not in files:
            raise FileNotFoundError(f"{WEIGHTS_FILE} must be in {path}.")

        with open(os.path.join(path, CONFIG_JSON_FILE), "r") as f:
            self.config = json.load(f)
        self.tokenizer = AlbertTokenizer.from_pretrained(path)
        self.weights = torch.load(os.path.join(path, WEIGHTS_FILE),
                                  map_location=lambda storage, loc: storage)

    def save(self, path, exclude=None):
        """ Method for model saving. """
        if not os.path.exists(path):
            os.mkdir(path)

        if type(exclude) is not list:
            exclude = [list]

        if not CONFIG_JSON_ID in exclude:
            if self.config is None:
                raise ValueError("Model have not been initialized successfully. Config missing.")
            with open(os.path.join(path, CONFIG_JSON_FILE), "w") as f:
                json.dump(self.config, f)
        if not TOKENIZER_ID in exclude:
            if self.tokenizer is None:
                raise ValueError("Model have not been initialized successfully. tokenizer missing.")
            self.tokenizer.save_pretrained(path)
        if not WEIGHTS_ID in exclude:
            if self.weights is None:
                raise ValueError("Model have not been initialized successfully. weights missing.")
            torch.save(self.weights, os.path.join(path, WEIGHTS_FILE))

    def _to_list(self, tensor):
        return tensor.detach().cpu().tolist()

    def _evaluate(self, args, model, tokenizer):
        dataset, examples, features = self._load_and_cache_examples(args, tokenizer, evaluate=True,
                                                                    output_examples=True)

        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu evaluate
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_results = []

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                example_indices = batch[3]

                outputs = model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [self._to_list(output[i]) for output in outputs]

                # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
                # models only use two.
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)

        output_prediction_file = os.path.join(args.output_dir, "predictions.json")
        output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds.json")

        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
        )

        os.remove(output_prediction_file)
        os.remove(output_nbest_file)
        os.remove(output_null_log_odds_file)

        return predictions

    def _load_and_cache_examples(self, args, tokenizer, evaluate=False, output_examples=False):
        if args.local_rank not in [-1, 0] and not evaluate:
            # Make sure only the first process in distributed training process the dataset, and the others will use the cache
            torch.distributed.barrier()

        # Load data features from cache or dataset file
        input_dir = args.data_dir if args.data_dir else "."
        cached_features_file = os.path.join(
            input_dir,
            "cached_{}_{}_{}".format(
                "dev" if evaluate else "train",
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
            ),
        )

        # Init features and dataset from cache if it exists
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features_and_dataset = torch.load(cached_features_file)
            features, dataset, examples = (
                features_and_dataset["features"],
                features_and_dataset["dataset"],
                features_and_dataset["examples"],
            )
        else:
            processor = SquadV2Processor()
            examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)

            features, dataset = squad_convert_examples_to_features(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=not evaluate,
                return_dataset="pt",
                threads=args.threads,
            )

            if args.local_rank in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

        if args.local_rank == 0 and not evaluate:
            # Make sure only the first process in distributed training process the dataset, and the others will use the cache
            torch.distributed.barrier()

        if evaluate:
            os.remove(cached_features_file)

        if output_examples:
            return dataset, examples, features
        return dataset

    def answer(self, question, context, **kwargs):
        device = kwargs.get('device', 'cpu')
        
        args = albert_args_squad
        for key in kwargs:
            if key in args:
                args[key] = kwargs[key]

        # Load pretrained model/tokenizer
        config = AlbertConfig.from_dict(self.config)
        model = AlbertForQuestionAnswering(config)
        model.load_state_dict(self.weights)
        model = model.eval()
        if device == "cuda":
            logger.debug("Setting model with CUDA")
            model.to('cuda')

        args = SimpleNamespace(**args)
        args.predict_file = "tmp.json"
        args.output_dir = "."

        predict_file = {
            'data': [{
                'title': "prediction",
                'paragraphs': [{
                    'qas': [{
                        'question': question,
                        'id': 0,
                        'answers': {}
                    }],
                    'context': context,
                    'is_impossible': False
                }]
            }]
        }
        with open(args.predict_file, "w") as f:
            json.dump(predict_file, f)

        result = self._evaluate(args, model, self.tokenizer)
        result = result[0]

        os.remove(args.predict_file)

        return result


class AlbertNER:
    def __init__(self):
        print("--- Model Albert for NER ---")
        
    def load(self, path):
        """ Method for pretrained model loading. """
        if not os.path.exists(path):
            raise NotADirectoryError(
                f"{os.path.abspath(path)} must be a directory containing the model files: config, tokenizer, weights.")

        files = os.listdir(path)
        if CONFIG_JSON_FILE not in files:
            raise FileNotFoundError(f"{CONFIG_JSON_FILE} must be in {path}.")
        if WEIGHTS_FILE not in files:
            raise FileNotFoundError(f"{WEIGHTS_FILE} must be in {path}.")

        with open(os.path.join(path, CONFIG_JSON_FILE), "r") as f:
            self.config = json.load(f)
        self.tokenizer = AlbertTokenizer.from_pretrained(path)
        self.weights = torch.load(os.path.join(path, WEIGHTS_FILE),
                                  map_location=lambda storage, loc: storage)

    def save(self, path, exclude=None):
        """ Method for model saving. """
        if not os.path.exists(path):
            os.mkdir(path)

        if type(exclude) is not list:
            exclude = [list]

        if not CONFIG_JSON_ID in exclude:
            if self.config is None:
                raise ValueError("Model have not been initialized successfully. Config missing.")
            with open(os.path.join(path, CONFIG_JSON_FILE), "w") as f:
                json.dump(self.config, f)
        if not TOKENIZER_ID in exclude:
            if self.tokenizer is None:
                raise ValueError("Model have not been initialized successfully. tokenizer missing.")
            self.tokenizer.save_pretrained(path)
        if not WEIGHTS_ID in exclude:
            if self.weights is None:
                raise ValueError("Model have not been initialized successfully. weights missing.")
            torch.save(self.weights, os.path.join(path, WEIGHTS_FILE))

    @staticmethod
    def set_seed(args):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

    def _train(self, args, train_dataset, model, tokenizer, labels, pad_token_label_id):
        """ Train the model """
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()

        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
                os.path.join(args.model_name_or_path, "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
            )

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.train_batch_size
            * args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
        )
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if os.path.exists(args.model_name_or_path):
            # set global_step to gobal_step of last saved checkpoint from model path
            try:
                global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
            except ValueError:
                global_step = 0
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
        )
        self.set_seed(args)  # Added here for reproductibility
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet"] else None
                    )  # XLM and RoBERTa don"t use segment_ids

                outputs = model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
                        if (
                                args.local_rank == -1 and args.evaluate_during_training
                        ):  # Only evaluate when single GPU otherwise metrics may not average well
                            results, _ = self._evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="train")
                            for key, value in results.items():
                                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                        logging_loss = tr_loss

                    if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        # model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        torch.save(model.state_dict(), os.path.join(output_dir, WEIGHTS_FILE))
                        model.config.to_json_file(os.path.join(output_dir, CONFIG_JSON_FILE))

                        # torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break
            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break

        if args.local_rank in [-1, 0]:
            tb_writer.close()

        return global_step, tr_loss / global_step

    def _evaluate(self, args, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
        eval_dataset = self._load_and_cache_examples(args, tokenizer, labels, pad_token_label_id,
                                                     mode=mode, evaluate=True)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu evaluate
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation %s *****", prefix)
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                if args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

                eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=2)

        label_map = {i: label for i, label in enumerate(labels)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        results = {
            "loss": eval_loss,
            "precision": ner_precision_score(out_label_list, preds_list),
            "recall": ner_recall_score(out_label_list, preds_list),
            "f1": ner_f1_score(out_label_list, preds_list),
        }

        logger.info("***** Eval results %s *****", prefix)
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results, preds_list

    def _load_and_cache_examples(self, args, tokenizer, labels, pad_token_label_id, mode, evaluate):
        if args.local_rank not in [-1, 0] and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}".format(
                mode, "albert", str(args.max_seq_length)
            ),
        )
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", args.data_dir)
            examples = read_examples_from_file(args.data_dir, mode)
            features = convert_examples_to_features(
                examples,
                labels,
                args.max_seq_length,
                tokenizer,
                cls_token_at_end=False,
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=0,
                sep_token=tokenizer.sep_token,
                sep_token_extra=False,
                pad_on_left=False,
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=0,
                pad_token_label_id=pad_token_label_id,
            )
            if args.local_rank in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

        if args.local_rank == 0 and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the
            # dataset, and the others will use the cache

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return dataset

    def extract(self, text, **kwargs):
        device = kwargs.get('device', 'cpu')
        data_dir = ""
        args = albert_args
        args['data_dir'] = data_dir
        for key in kwargs:
            if key in args:
                args[key] = kwargs[key]

        # Load pretrained model/tokenizer
        config = AlbertConfig.from_dict(self.config)
        model = AlbertForTokenClassification(config)
        model.load_state_dict(self.weights)
        model = model.eval()
        if device == "cuda":
            logger.debug("Setting model with CUDA")
            model.to('cuda')

        args = SimpleNamespace(**args)
        labels = [x[1] for x in self.config['id2label'].items()]
        num_labels = len(labels)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = CrossEntropyLoss().ignore_index

        text = tokenize_sentence_wbl(text.replace("\n", " "))
        with open(f"test.txt", "w") as f:
            f.write(text)
        result, predictions = self._evaluate(args, model, self.tokenizer, labels, pad_token_label_id, mode="test")
        os.remove("test.txt")

        return parse_result(text, predictions)

    def train(self, data_path, **kwargs):
        train_steps = kwargs.get('train_steps', 1000)
        batch_size = kwargs.get('batch_size', 8)
        device = kwargs.get('device', None)

        args = albert_args
        args['data_dir'] = data_path
        args['max_steps'] = train_steps
        args['per_gpu_train_batch_size'] = batch_size
        if device == 'cpu':
            args['no_cuda'] = True
        else:
            args['no_cuda'] = False
        for key in kwargs:
            if key in args:
                args[key] = kwargs[key]

        args = SimpleNamespace(**args)

        if args.local_rank == -1 or args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            args.n_gpu = 1

        # Load pretrained model/tokenizer
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        if self.weights and self.config and self.tokenizer:
            config = AlbertConfig.from_dict(self.config)
            model = AlbertForTokenClassification(config)
            model.load_state_dict(self.weights)
            tokenizer = self.tokenizer
        else:
            config = AlbertConfig()
            model = AlbertForTokenClassification(config)
            tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')

        if args.local_rank == 0:
            torch.distributed.barrier()

        model.to(device)

        # Set seed
        self.set_seed(args)

        labels = get_labels(args.labels)
        num_labels = len(labels)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = CrossEntropyLoss().ignore_index

        train_dataset = self._load_and_cache_examples(args, tokenizer, labels, pad_token_label_id,
                                                      mode="train", evaluate=False)
        global_step, tr_loss = self._train(args, train_dataset, model, tokenizer, labels, pad_token_label_id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        self.weights = model.state_dict()
