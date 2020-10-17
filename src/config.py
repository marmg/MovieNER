CONFIG_JSON_FILE = "config.json"
TOKENIZER_FILE = "tokenizer.pkl"
WEIGHTS_FILE = "checkpoint.pt"

CONFIG_JSON_ID = "config"
TOKENIZER_ID = "tokenizer"
WEIGHTS_ID = "weights"

albert_args = {
    "data_dir": "",
    # "The input data dir. Should contain the training files for the CoNLL-2003 NER task."
    "model_type": "albert",
    # "Model type selected in the list": " + ", ".join(MODEL_TYPES),
    "model_name_or_path": "albert",
    # "Path to pre-trained model or shortcut name selected in the list": " + ", ".join(ALL_MODELS),
    "output_dir": "",
    # "The output directory where the model predictions and checkpoints will be written.",
    "labels": "",
    # "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
    "config_name": "",
    # "Pretrained config name or path if not the same as model_name",
    "tokenizer_name": "",
    # "Pretrained tokenizer name or path if not the same as model_name",
    "cache_dir": "",
    # "Where do you want to store the pre-trained models downloaded from s3",
    "max_seq_length": 128,
    # "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.",
    "evaluate_during_training": True,
    # "Whether to run evaluation during training at each logging step.",
    "n_gpu": 0,
    "per_gpu_train_batch_size": 8,
    # "Batch size per GPU/CPU for training.",
    "per_gpu_eval_batch_size": 8,
    # "Batch size per GPU/CPU for evaluation.",
    "gradient_accumulation_steps": 1,
    # "Number of updates steps to accumulate before performing a backward/update pass.",
    "learning_rate": 5e-5,
    # "The initial learning rate for Adam.",
    "weight_decay": 0.0,
    # "Weight decay if we apply some.",
    "adam_epsilon": 1e-8,
    # "Epsilon for Adam optimizer.",
    "max_grad_norm": 1.0,
    # "Max gradient norm.",
    "num_train_epochs": 3.0,
    # "Total number of training epochs to perform.",
    "max_steps": -1,
    # "If > 0: set total number of training steps to perform. Override num_train_epochs.",
    "warmup_steps": 0,
    # "Linear warmup over warmup_steps."
    "logging_steps": 500,
    # "Log every X updates steps."
    "save_steps": 500,
    # "Save checkpoint every X updates steps."
    "eval_all_checkpoints": True,
    # "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    "seed": 42,
    # "random seed for initialization"
    "fp16": False,
    "fp16_opt_level": "O1",
    # "For fp16": Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details at https"://nvidia.github.io/apex/amp.html",
    "local_rank": -1,
    # "For distributed training": local_rank"
    "overwrite_cache": False,
    "overwrite_output_dir": False,
    "device": "cpu"
}

albert_args_squad = {
    "data_dir": None,
    # "The input data dir. Should contain the training files for the CoNLL-2003 NER task."
    "model_type": "albert",
    # "Model type selected in the list": " + ", ".join(MODEL_TYPES),
    "model_name_or_path": "albert",
    # "Path to pre-trained model or shortcut name selected in the list": " + ", ".join(ALL_MODELS),
    "output_dir": "",
    # "The output directory where the model predictions and checkpoints will be written.",
    "train_file": None,
    #The input training file. If a data dir is specified, will look for the file there. If no data dir or train/predict files are specified, will run with tensorflow_datasets.
    "predict_file": "",
    # The input evaluation file. If a data dir is specified, will look for the file there. If no data dir or train/predict files are specified, will run with tensorflow_datasets.
    "config_name": "",
    # "Pretrained config name or path if not the same as model_name",
    "tokenizer_name": "",
    # "Pretrained tokenizer name or path if not the same as model_name",
    "cache_dir": "",
    # "Where do you want to store the pre-trained models downloaded from s3",
    "max_seq_length": 384,
    # "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.",
    "doc_stride": 128,
    "do_lower_case": False,
    # When splitting up a long document into chunks, how much stride to take between chunks.
    "max_query_length": 64,
    # The maximum number of tokens for the question. Questions longer than this will be truncated to this length.
    "evaluate_during_training": True,
    # "Whether to run evaluation during training at each logging step.",
    "n_gpu": 0,
    "per_gpu_train_batch_size": 8,
    # "Batch size per GPU/CPU for training.",
    "per_gpu_eval_batch_size": 8,
    # "Batch size per GPU/CPU for evaluation.",
    "gradient_accumulation_steps": 1,
    # "Number of updates steps to accumulate before performing a backward/update pass.",
    "learning_rate": 5e-5,
    # "The initial learning rate for Adam.",
    "weight_decay": 0.0,
    # "Weight decay if we apply some.",
    "adam_epsilon": 1e-8,
    # "Epsilon for Adam optimizer.",
    "max_grad_norm": 1.0,
    # "Max gradient norm.",
    "num_train_epochs": 3.0,
    # "Total number of training epochs to perform.",
    "max_steps": -1,
    # "If > 0: set total number of training steps to perform. Override num_train_epochs.",
    "warmup_steps": 0,
    # "Linear warmup over warmup_steps."
    "logging_steps": 500,
    # "Log every X updates steps."
    "save_steps": 500,
    # "Save checkpoint every X updates steps."
    "eval_all_checkpoints": True,
    # "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    "seed": 42,
    # "random seed for initialization"
    "fp16_opt_level": "O1",
    # "For fp16": Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details at https"://nvidia.github.io/apex/amp.html",
    "local_rank": -1,
    # "For distributed training": local_rank"
    "overwrite_cache": True,
    "overwrite_output_dir": False,
    "device": "cpu",
    "version_2_with_negative": True,
    # Squad_v2
    "null_score_diff_threshold": 0.0,
    # If null_score - best_non_null is greater than the threshold predict null
    "n_best_size": 20,
    # The total number of n-best predictions to generate in the nbest_predictions.json output file.
    "max_answer_length": 30,
    # The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.
    "threads": 1,
    # multiple threads for converting example to features
    "verbose_logging": False
}

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']

model_name = "model_step_i.pt"
summary_fn = ".-1.candidate"
