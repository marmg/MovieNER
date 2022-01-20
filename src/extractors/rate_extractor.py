import re
import string
from typing import List

from src.extractors.base_extractor import BaseExtractor

# Regex
rates_l = [
    "g",
    "m",
    "gp",
    "pg\-7",
    "pg\-12",
    "pg\-13",
    "pg\-16",
    "pg\-18",
    "pg\-11",
    "pg 7",
    "pg 11",
    "pg 12",
    "pg 13",
    "pg 16",
    "pg 18",
    "pg\+7",
    "pg\+11",
    "pg\+12",
    "pg\+13",
    "pg\+16",
    "pg\+18",
    "pg",
    "r",
    "x",
    "nc\-17",
    "nr",
    "ur",
    "\+7",
    "\+12",
    "\+18",
    "General Audiences",
    "Parental Guidance Suggested",
    "Parents Strongly Cautioned",
    "Restricted",
    "Adults Only"
]
pat_rate = fr"\b(?:{'|'.join(rates_l)})\b(?:-rated| rated|)"


class RateExtractor(BaseExtractor):
    """ Extract rates """

    def get_rate_avg(self, **kwargs: dict) -> List[str]:
        """ Get average rate """
        ratings_avg = [ent[0] for ent in kwargs['entities_spacy'] if ent[1] == "CARDINAL"]
        ratings_avg = list(set([t for t in ratings_avg]))

        real_ratings = []
        text_words = kwargs['text'].split()
        for rat in ratings_avg:
            w = rat.split()[-1]
            i = text_words.index(w)
            if len(text_words) > i + 1:
                if text_words[i + 1].strip(string.punctuation) == "stars":
                    real_ratings.append(rat + " " + "stars")
            if "give" in text_words[i - 5:i + 5] or "/" in rat:
                real_ratings.append(rat)

        return real_ratings

    def get_rate(self, **kwargs: dict) -> List[str]:
        """ Get rates from text. Will take text from kwargs and will return the rates extracted """
        rates = re.findall(pat_rate, kwargs['text'], re.IGNORECASE)

        return rates

    def run(self, **kwargs: dict) -> dict:
        """ Execute extractor. Will update kwargs with the rates extracted """
        kwargs['rate_avg'] = self.get_rate_avg(**kwargs)
        kwargs['rate'] = self.get_rate(**kwargs)
        return kwargs
