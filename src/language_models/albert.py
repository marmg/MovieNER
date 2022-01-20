import argparse
import glob
import json
import logging
import numpy as np
import os
import pickle as pkl
import random
import torch
from tqdm import tqdm, trange
from transformers import AlbertConfig, AlbertForTokenClassification, AutoTokenizer, AlbertForQuestionAnswering
from typing import List, Tuple

from src.language_models.config import *
from src.logging_handler import logger


class AlbertQA:
    """ Class to use Albert to answer questions.
    TODO: Update model and checkpoints to work with last versions of transformers """

    def __init__(self, path: str, device: str = 'cpu'):
        """ Init the QA Albert """
        if not os.path.exists(path):
            raise NotADirectoryError(
                f"{os.path.abspath(path)} must be a directory containing the model files: config, tokenizer, weights.")

        files = os.listdir(path)
        if CONFIG_JSON_FILE not in files:
            raise FileNotFoundError(f"{CONFIG_JSON_FILE} must be in {path}.")
        if WEIGHTS_FILE not in files:
            raise FileNotFoundError(f"{WEIGHTS_FILE} must be in {path}.")

        with open(os.path.join(path, CONFIG_JSON_FILE), "r") as f:
            config = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        weights = torch.load(os.path.join(path, WEIGHTS_FILE),
                                  map_location=lambda storage, loc: storage)
        # Load pretrained model/tokenizer
        config = AlbertConfig.from_dict(config)
        self.model = AlbertForQuestionAnswering(config)
        self.model.load_state_dict(weights)
        self.model = self.model.eval()
        self.args = albert_args_squad
        if device == "cuda":
            logger.debug("Setting model with CUDA")
            self.args['device'] = 'cuda'
            self.model.to('cuda')

    def answer(self, question: str, context: str, **kwargs: dict) -> str:
        """ Look the answer to question in context

        Keyword Arguments:
             :param question: Question to answer
             :param context: Context to look for the answer into
             :return: Answer to question
        """
        for key in kwargs:
            if key in self.args:
                self.args[key] = kwargs[key]
        inputs = self.tokenizer.encode_plus(question, context, **self.args)
        for key in inputs.keys():
            inputs[key] = inputs[key].to(self.args['device'])
        input_ids = inputs["input_ids"].tolist()[0]

        answer_start_scores, answer_end_scores = self.model(**inputs)

        answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer
        answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer

        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(
                input_ids[answer_start:answer_end]
            )
        )
        answer = answer.replace("[CLS]", "").replace("[SEP]", " ").replace("<s>", "").replace("</s>", "")
        return answer


class AlbertNER:
    """ Class to use Albert to extract named entities.
    TODO: Update model and checkpoints to work with last versions of transformers """
    def __init__(self, path: str ,device: str = 'cpu'):
        """ Init the NER Albert """
        if not os.path.exists(path):
            raise NotADirectoryError(
                f"{os.path.abspath(path)} must be a directory containing the model files: config, tokenizer, weights.")

        files = os.listdir(path)
        if CONFIG_JSON_FILE not in files:
            raise FileNotFoundError(f"{CONFIG_JSON_FILE} must be in {path}.")
        if WEIGHTS_FILE not in files:
            raise FileNotFoundError(f"{WEIGHTS_FILE} must be in {path}.")

        with open(os.path.join(path, CONFIG_JSON_FILE), "r") as f:
            config = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        weights = torch.load(os.path.join(path, WEIGHTS_FILE),
                                  map_location=lambda storage, loc: storage)
        # Load pretrained model/tokenizer
        config = AlbertConfig.from_dict(config)
        self.model = AlbertForTokenClassification(config)
        self.model.load_state_dict(weights)
        self.model = self.model.eval()
        self.args = albert_args_ner
        if device == "cuda":
            logger.debug("Setting model with CUDA")
            self.args['device'] = 'cuda'
            self.model.to('cuda')


    def extract(self, text: str, **kwargs: dict) -> List[Tuple[str, str]]:
        """ Extract named entities from text

        Keyword Arguments:
            :param text: Text to extract entities from
            :return: List of named entities extacted
        """
        for key in kwargs:
    	    if key in self.args:
                self.args[key] = kwargs[key]

        tokens = self.tokenizer.tokenize(self.tokenizer.decode(self.tokenizer.encode(text)))
        inputs = self.tokenizer.encode(text, return_tensors="pt")

        outputs = self.model(inputs, **kwargs)[0]
        predictions = torch.argmax(outputs, dim=2)

        return [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].tolist())]
