import re


class GenreExtractor(BaseExtractor):
	def get_genre(self):
		genres_tmp = []
		for genre in genres:
			if genre in self.text:
				genres_tmp.append(genre)
		
		return genres_tmp

	def get_genres_from_df(self, genre):
		genres = genre.split(",")
		pat = fr"\b(?:{'|'.join(genres)})\b"
		genres = re.findall(pat, self.text, re.IGNORECASE)
		
		return genres
			
	def run(self, **kwargs):
		kwargs['genres'] = self.get_genre()
		return kwargs