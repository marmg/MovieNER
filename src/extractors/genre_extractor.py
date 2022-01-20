import re
import os

from src.extractors.base_extractor import BaseExtractor
from src.endpoint_config import ASSETS_PATH

with open(os.path.join(ASSETS_PATH, "genres.list"), "r") as f:
    GENRES = f.read().split("\n")


class GenreExtractor(BaseExtractor):
	def get_genre(self, **kwargs):
		genres_tmp = []
		for genre in GENRES:
			if genre in kwargs['text']:
				genres_tmp.append(genre)
		
		return genres_tmp

	def get_genres_from_df(self, text, genre):
		genres = genre.split(",")
		pat = fr"\b(?:{'|'.join(genres)})\b"
		genres = re.findall(pat, text, re.IGNORECASE)
		
		return genres
			
	def run(self, **kwargs):
		kwargs['genres'] = self.get_genre(**kwargs)
		return kwargs