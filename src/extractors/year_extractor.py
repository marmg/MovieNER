import re

from src.extractors.base_extractor import BaseExtractor

pat_year = "(?:since|before|after|)(?:1[8-9]{1}[0-9]{2}|2[0-9]{3})'?s?"

class YearExtractor(BaseExtractor):
	def get_year(self, **kwargs):
		dates = re.findall(pat_year, kwargs['text'], re.IGNORECASE)
		dates = list(set([t for t in dates]))
		
		return dates

	def run(self, **kwargs):
		kwargs['years'] = self.get_year(**kwargs)
		return kwargs