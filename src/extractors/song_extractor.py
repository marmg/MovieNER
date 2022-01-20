import re
from typing import List, Tuple

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
    """ Extract songs. """
    def get_songs(self, **kwargs: dict) -> Tuple[List[str], List[str]]:
        """ Get songs from text and update characters taking out the compositors.
        Will take text and characters from kwargs and will return songs extracted and list of characters updated """
        characters = kwargs.get('characters', [])
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
                text_tmp = kwargs['text'][text_idx_tmp - 50:text_idx_tmp + len(song) + 50]
                if character in text_tmp and re.findall("(?:by |of |from )", text_tmp):
                    add = False
            if add:
                new_characters.append(character)
            else:
                songs.append(character)

        return songs, new_characters

    def run(self, **kwargs: dict) -> dict:
        """ Execute extractor. Will update kwargs with the songs extracted """
        songs, characters = self.get_songs(**kwargs)
        kwargs['songs'] = songs
        kwargs['characters'] = characters

        return kwargs
