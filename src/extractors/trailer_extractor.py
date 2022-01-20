import re

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
	def get_trailers(self, **kwargs):
		trailers = re.findall(pat_trailers, kwargs['text'], re.IGNORECASE)
		
		return trailers

	def run(self, **kwargs):
		kwargs['trailers'] = self.get_awards(**kwargs)
		
		return kwargs