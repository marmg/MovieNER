import re
from typing import List

from src.extractors.base_extractor import BaseExtractor


trailers_l = [
   "trailer",
   "trailers",
   "corto",
   "cut",
   "advance",
   "announce"
]

pat_trailers = fr"\b(?:{'|'.join(trailers_l)})\b"


class TrailerExtractor(BaseExtractor):
    """ Extract Trailers """
	def get_trailers(self, **kwargs: dict) -> List[str]:
        """ Get trailers from text. Will take text from kwargs and will return trailers extracted"""
		trailers = re.findall(pat_trailers, kwargs['text'], re.IGNORECASE)
		
		return trailers

	def run(self, **kwargs: dict) -> dict:
        """ Execute extractor. Will update kwargs with the trailers extracted """
		kwargs['trailers'] = self.get_trailers(**kwargs)
		
		return kwargs