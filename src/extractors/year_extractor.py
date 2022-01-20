import re
from typing import List

from src.extractors.base_extractor import BaseExtractor

pat_year = "(?:since|before|after|)(?:1[8-9]{1}[0-9]{2}|2[0-9]{3})'?s?"

class YearExtractor(BaseExtractor):
	""" Extract Years """
	def get_year(self, **kwargs: dict) -> List[str]:
		""" Get year from text. Will take text from kwargs and will return dates extracted """
		dates = re.findall(pat_year, kwargs['text'], re.IGNORECASE)
		dates = list(set([t for t in dates]))
		
		return dates

	def run(self, **kwargs: dict) -> dict:
        """ Execute extractor. Will update kwargs with the years extracted """
		kwargs['years'] = self.get_year(**kwargs)
		return kwargs