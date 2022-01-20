# Imports
import spacy
import pandas as pd

from src.language_models.albert import AlbertNER
from src.extractors import *
from src.checker import Checker
from src.filter import Filter

from .endpoint_config import ASSETS_PATH, MODELS_PATH
from .logging_handler import init_logger, logger

import logging
import string
import re
import os


ACTOR = "actor"
DIRECTOR = "director"
CHARACETR = "character"


class Detector:
    def __init__(self):
        # NER Models
        self. model = spacy.load("en_core_web_lg")
        self.ner = AlbertNER(os.path.join(MODELS_PATH, "conll03"))

        # Check data with movie database
        cols = ["original_title", "year", "genre", "director", "actors", "description"]
        df_movies = pd.read_csv(os.path.join(ASSETS_PATH, "movies.csv"))
        self.df_movies = df_movies.loc[df_movies.actors.notna()]

        # Extractors
        self.award_extractor = AwardsExtractor(df_movies)
        self.genre_extractor = GenreExtractor(df_movies)
        self.person_extractor = PersonExtractor(df_movies)
        self.rate_extractor = RateExtractor(df_movies)
        self.song_extractor = SongExtractor(df_movies)
        self.title_extractor = TitleExtractor(df_movies)
        self.trailer_extractor = TrailerExtractor(df_movies)
        self.year_extractor = YearExtractor(df_movies)
        self.extractors = [
            self.award_extractor,
            self.genre_extractor,
            self.person_extractor,
            self.rate_extractor,
            self.song_extractor,
            self.title_extractor,
            self.trailer_extractor,
            self.year_extractor
        ]

        # Filter
        self.filter = Filter(df_movies)
        # Checker
        self.checker = Checker(self.filter, df_movies)

    def get_entities(self, **kwargs):
        doc = self.model(kwargs['text'])
        kwargs['entities_spacy'] = [(ent.text, ent.label_) for ent in doc.ents]
        kwargs['entities_albert'] = self.ner.extract(kwargs['text'])

        return kwargs

    def parse_entity(self, entity_text, label):
        words = entity_text.split(" ")
        entities = [(words[0], f"B-{label}")]
        entities += [(w, f"I-{label}") for w in words[1:]]

        return entities

    def merge_entities(self, entities, new_entities):
        original_words = list(enumerate([ent_[0].strip().strip(string.punctuation) for ent_ in entities]))
        for ent in new_entities:
            words = [x[0] for x in ent]
            idxs = []
            for i, word in original_words:
                if word == words[0].strip().strip(string.punctuation):
                    val = True
                    for j in range(len(words)):
                        if entities[i+j][0] != words[j].strip().strip(string.punctuation):
                            val = False
                            break
                    if val:
                        idxs += list(zip(ent, range(i, i+len(words))))
            for ent_, i in idxs:
                if ent_[1].startswith("I"):
                    if i != 0 and (entities[i-1][1] == ent_[1].replace("I-", "B-") or entities[i-1][1] == ent_[1]):
                        entities[i] = ent_
                else:
                    entities[i] = ent_

        return entities

    def parse_entities(self, **kwargs):
        titles_parsed = []
        for title in kwargs['titles']:
            titles_parsed.append(self.parse_entity(title.strip(), "TITLE"))
        years_parsed = []
        for year in kwargs['years']:
            years_parsed.append(self.parse_entity(year.strip(), "YEAR"))
        ratings_avg_parsed = []
        for rating_average in kwargs['rate_avg']:
            ratings_avg_parsed.append(self.parse_entity(rating_average.strip(), "RATINGS_AVERAGE"))
        awards_parsed = []
        for award in kwargs['awards']:
            awards_parsed.append(self.parse_entity(award.strip(), "AWARD"))
        songs_parsed = []
        for song in kwargs['songs']:
            songs_parsed.append(self.parse_entity(song.strip(), "SONG"))
        trailers_parsed = []
        for trailer in kwargs['trailers']:
            trailers_parsed.append(self.parse_entity(trailer.strip(), "TRAILER"))
        rate_parsed = []
        for rating in kwargs['rate']:
            rate_parsed.append(self.parse_entity(rating.strip(), "RATING"))
        genres_parsed = []
        for genre in kwargs['genres']:
            genres_parsed.append(self.parse_entity(genre.strip(), "GENRE"))
        actors_parsed = []
        for actor in kwargs['actors']:
            actors_parsed.append(self.parse_entity(actor.strip(), "ACTOR"))
        directors_parsed = []
        for director in kwargs['directors']:
            directors_parsed.append(self.parse_entity(director.strip(), "DIRECTOR"))
        characters_parsed = []
        for character in kwargs['characters']:
            characters_parsed.append(self.parse_entity(character.strip(), "CHARACTER"))
        new_entities = titles_parsed + years_parsed + ratings_avg_parsed + trailers_parsed + \
                       rate_parsed + genres_parsed + directors_parsed + actors_parsed + \
                       characters_parsed + songs_parsed + awards_parsed
        return new_entities


    def extract(self, text):
        kwargs = {'text': text}
        words = text.split(" ")
        entities = [(w.strip().strip(string.punctuation), "O") for w in words]

        kwargs = self.get_entities(**kwargs)
        for extractor in self.extractors:
            kwargs = extractor.run(**kwarg)
        kwargs, df = self.checker.run(**kwargs)

        if len(df) == 1:
            kwargs['titles'] = self.title_extractor.get_titles_from_df(text, df.original_title.values[0])
            kwargs['genres'] = self.genre_extractor.get_genres_from_df(text, df.genre.values[0])
            kwargs['actors'] = list(set(actors+self.person_extractor.get_actors_from_df(text, df.actors.values[0])))
            kwargs['directors'] = self.person_extractor.get_directors_from_df(text, df.director.values[0])

        new_entities = self.parse_entities(**kwargs)
        entities = self.merge_entities(entities, new_entities)
        if len(df) == 1:
            return entities, df
        else:
            return entities, None
