import re


class TitleExtractor(BaseExtractor):		
	def get_titles(self, **kwargs):
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
		titles_tmp = [t for t in titles_tmp if t.lower() in titles]
		
		return titles_tmp

	def get_titles_from_re(self, **kwargs):
		text_words = kwargs['text'].split()
		if "film" in text_words:
			i = text_words.index("film")
			for j in reversed(range(4)):
				ent_tmp = " ".join(text_words[i+1:i+j+1])
				if ent_tmp in titles:
					return [ent_tmp]
		
		if "movie" in text_words:
			i = text_words.index("movie")
			for j in reversed(range(5)):
				ent_tmp = " ".join(text_words[i:i+j])
				if ent_tmp in titles:
					return [ent_tmp]
				
		return []
				
	def get_titles_from_df(self, text, original_title):
		pat = fr"\b{original_title}\b"
		title = re.findall(pat, text, re.IGNORECASE)
		
		return title
		
	def run(self, **kwargs):
		kwargs['titles'] =  self.get_titles(**kwargs) + self.get_titles_from_re(**kwargs)
		return kwargs