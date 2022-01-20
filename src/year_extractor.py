import re


class YearExtractor(BaseExtractor):
	def get_year(self):
		dates = re.findall(pat_year, self.text, re.IGNORECASE)
		dates = list(set([t for t in dates]))
		
		return dates

	def run(self, **kwargs):
		kwargs['years'] = self.get_year()
		return kwargs