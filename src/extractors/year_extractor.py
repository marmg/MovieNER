import re


class YearExtractor(BaseExtractor):
	def get_year(self, **kwargs):
		dates = re.findall(pat_year, kwargs['text'], re.IGNORECASE)
		dates = list(set([t for t in dates]))
		
		return dates

	def run(self, **kwargs):
		kwargs['years'] = self.get_year(**kwargs)
		return kwargs