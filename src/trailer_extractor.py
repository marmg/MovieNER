import re


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
	def get_trailers(self):
		trailers = re.findall(pat_trailers, self.text, re.IGNORECASE)
		
		return trailers

	def run(self, **kwargs):
		kwargs['trailers'] = self.get_awards()
		
		return kwargs