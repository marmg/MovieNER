import re
import os
from typing import List

from src.extractors.base_extractor import BaseExtractor
from src.endpoint_config import ASSETS_PATH

with open(os.path.join(ASSETS_PATH, "genres.list"), "r") as f:
    GENRES = f.read().split("\n")


class GenreExtractor(BaseExtractor):
    """ Extract Genres """
    def get_genre(self, **kwargs: dict) -> List[str]:
        """ Get genres from text. Will take text from kwargs and will return genres extracted """
        genres_tmp = []
        for genre in GENRES:
            if genre in kwargs['text']:
                genres_tmp.append(genre)

        return genres_tmp

    def get_genres_from_df(self, text: str, genre: str) -> List[str]:
        """ Get genres from text.

        Keyword Arguments:
            :param text: Text to extract genres from
            :param genre: List of genres to look for in text
            :return: Genres extarcted
        """
        genres = genre.split(",")
        pat = fr"\b(?:{'|'.join(genres)})\b"
        genres = re.findall(pat, text, re.IGNORECASE)

        return genres

    def run(self, **kwargs: dict) -> dict:
        """ Execute extractor. Will update kwargs with the genres extracted """
        kwargs['genres'] = self.get_genre(**kwargs)
        return kwargs
