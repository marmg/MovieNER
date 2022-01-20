import re

from src.extractors.base_extractor import BaseExtractor

awards_l = [
    "oscars",
    "sag",
    "sag award",
    "sag awards",
    "award",
    "awards",
    "Best Feature Film",
    "Best Actor",
    "Best Short Film",
    "Best British Short Film",
    "Special Jury Prize for Short Film",
    "Best Feature Documentary",
    "Best Student Film",
    "Best Music Video",
    "Best Short Documentary",
    "Best Animation",
    "Best First Film",
    "Best Picture",
    "Top Ten Films",
    "Best Director",
    "Best Actor",
    "Best Actress",
    "Best Supporting Actor",
    "Best Supporting Actress",
    "Best Ensemble Cast",
    "Best Original Screenplay",
    "Best Adapted Screenplay",
    "Best Cinematography",
    "Best Production Design",
    "Best Editing",
    "Best Original Score",
    "Best Visual Effects"
    "academy"
]

pat_awards = fr"\b(?:{'|'.join(awards_l)})\b"


class AwardsExtractor(BaseExtractor):
    """ Extract Awards """

    def get_awards(self, **kwargs: dict) -> List[str]:
        """ Get Awards from text. Will take text from kwargs """
        awards = re.findall(pat_awards, kwargs['text'], re.IGNORECASE)

        return awards

    def run(self, **kwargs: dict) -> dict:
        """ Execute extractor. Will update kwargs with the awards extracted """
        kwargs['awards'] = self.get_awards(**kwargs)

        return kwargs
