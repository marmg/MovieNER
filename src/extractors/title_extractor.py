import re
import os
from typing import List

from src.extractors.base_extractor import BaseExtractor
from src.endpoint_config import ASSETS_PATH

with open(os.path.join(ASSETS_PATH, "titles.list"), "r") as f:
    TITLES = f.read().split("\n")


class TitleExtractor(BaseExtractor):
	""" Extract titles """
	def get_titles(self, **kwargs: dict) -> List[str]:
		""" Get titles. Will take Albert entities from kwargs and will return titles extracted. """
		titles_tmp = []
		t = ""
		for ent in kwargs['entities_albert']:
			if ent[1] == "B-MISC":
				if t:
					titles_tmp.append(t)
				t = ent[0]
			elif ent[1] == "I-MISC":
				t += " " + ent[0]
			else:
				if t:
					titles_tmp.append(t)
				t = ""

		titles_tmp = list(set([t for t in titles_tmp]))
		titles_tmp = [t for t in titles_tmp if t.lower() in TITLES]
		
		return titles_tmp

	def get_titles_from_re(self, **kwargs: dict) -> List[str]:
		""" Get titles. Will take text from kwargs and will return titles extracted. """
		text_words = kwargs['text'].split()
		if "film" in text_words:
			i = text_words.index("film")
			for j in reversed(range(4)):
				ent_tmp = " ".join(text_words[i+1:i+j+1])
				if ent_tmp in TITLES:
					return [ent_tmp]
		
		if "movie" in text_words:
			i = text_words.index("movie")
			for j in reversed(range(5)):
				ent_tmp = " ".join(text_words[i:i+j])
				if ent_tmp in TITLES:
					return [ent_tmp]
				
		return []
				
	def get_titles_from_df(self, text: str, original_title: str) -> List[str]:
		""" Get titles from df.

		Keyword Arguments:
			:param text: Text to extract titles from
			:param original_title: Original title to look from in text
			:return: Title found
		"""
		pat = fr"\b{original_title}\b"
		title = re.findall(pat, text, re.IGNORECASE)
		
		return title
		
	def run(self, **kwargs: dict) -> dict:
		""" Execute extractor. Will update kwargs with the titles extracted """
		kwargs['titles'] =  self.get_titles(**kwargs) + self.get_titles_from_re(**kwargs)
		return kwargs