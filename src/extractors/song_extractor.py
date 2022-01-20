import re

from src.extractors.base_extractor import BaseExtractor


songs_l = [
   "song",
   "bso",
   "songs",
   "bsos",
   "music",
   "musical score",
   "track",
   "sound",
   "soundtrack",
   "soundtrac",
   "tracks",
   "soundtracks",
   "composition",
   "ost",
   "osts",
   "melody",
   "melodies",
   "lyric", 
   "lyrics", 
   "anthem",
   "anthems",
   "tune",
   "tunes",
   "sing",
   "piece", 
   "original soundtrack" 
]

pat_songs = fr"\b(?:{'|'.join(songs_l)})\b"


class SongExtractor(BaseExtractor):
	def get_songs(self, **kwargs):
		songs = re.findall(pat_songs, kwargs['text'], re.IGNORECASE)
		if not songs:
			return [], characters
		pat = fr"(?:{'|'.join(characters)}|)?\s?(?:{'|'.join([s.strip() for s in songs])})\s(?:by |of |from )?(?:{'|'.join(characters)}|)?"
		songs = re.findall(pat, kwargs['text'], re.IGNORECASE)
		new_characters = []
		for character in characters:
			add = True
			for song in songs:
				text_idx_tmp = kwargs['text'].index(song)
				text_tmp = kwargs['text'][text_idx_tmp-50:text_idx_tmp+len(song)+50]
				if character in text_tmp and re.findall("(?:by |of |from )", text_tmp):
					add = False
			if add:
				new_characters.append(character)
			else:
				songs.append(character)
		
		return songs, new_characters

	def run(self, **kwargs):
		songs, characters = self.get_songs(**kwargs)
		kwargs['songs'] = songs
		kwargs['characters'] = characters
		
		return kwargs