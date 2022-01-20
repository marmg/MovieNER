import re

from src.extractors.base_extractor import BaseExtractor


class GenreExtractor(BaseExtractor):
	def get_genre(self, **kwargs):
		genres_tmp = []
		for genre in genres:
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