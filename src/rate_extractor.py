import re
import string


# Regex
rates_l = [
   "g",
   "m",
   "gp",
   "pg\-7",
   "pg\-12",
   "pg\-13",
   "pg\-16",
   "pg\-18",
   "pg\-11",
   "pg 7",
   "pg 11",
   "pg 12",
   "pg 13",
   "pg 16",
   "pg 18",
   "pg\+7",
   "pg\+11",
   "pg\+12",
   "pg\+13",
   "pg\+16",
   "pg\+18",
   "pg",
   "r",
   "x",
   "nc\-17",
   "nr",
   "ur",
   "\+7",
   "\+12",
   "\+18",
   "General Audiences",
   "Parental Guidance Suggested",
   "Parents Strongly Cautioned",
   "Restricted",
   "Adults Only"
]
pat_rate = fr"\b(?:{'|'.join(rates_l)})\b(?:-rated| rated|)"

class RateExtractor(BaseExtractor):
			
	def get_rate_avg(self, entities_spacy):
		ratings_avg = [ent[0] for ent in entities_spacy if ent[1] == "CARDINAL"]
		ratings_avg = list(set([t for t in ratings_avg]))
		
		real_ratings = []
		text_words = self.text.split()
		for rat in ratings_avg:
			w = rat.split()[-1]
			i = text_words.index(w)
			if len(text_words) > i+1:
				if text_words[i+1].strip(string.punctuation) == "stars":
					real_ratings.append(rat + " " + "stars")
			if "give" in text_words[i-5:i+5] or "/" in rat:
				real_ratings.append(rat)
		
		return real_ratings

	def get_rate(self):
		rates = re.findall(pat_rate, self.text, re.IGNORECASE)
		
		return rates
		
	def run(self, **kwargs):
		kwargs['rate_avg'] = self.get_rate_avg(kwargs['entities_spacy'])
		kwargs['rate'] = self.get_rate()
		return kwargs