from src.language_models import AlbertQA
from src.endpoint_config import ASSETS_PATH, MODELS_PATH
from src.extractors.base_extractor import BaseExtractor

import string
import re
import os


ACTOR = "actor"
DIRECTOR = "director"
CHARACETR = "character"


# QA Model to disambiguate
qa = AlbertQA(os.path.join(MODELS_PATH, "squad"))


# List of actors and directors 
with open(os.path.join(ASSETS_PATH, "actors.list"), "r") as f:
    ACTORS = f.read().split("\n")
with open(os.path.join(ASSETS_PATH, "directors.list"), "r") as f:
    DIRECTORS = f.read().split("\n")
	
	
pat_director = "(?:directed|director|directed by)"


class PersonExtractor(BaseExtractor):
	""" Extract actors, directors and characters. """
	def get_persons(self, entities_spacy: List[Tuple[str, str]], entities_albert: List[Tuple[str, str]]):
		""" Get persons from extracted entities from NER models.

		Keyword Arguments:
			:param entities_spacy: Named Entities extracted from spacy model
			:param entities_albert: Named Entities extracted from Albert model
			:return: List of persons extracted
		"""
		persons = []
		for ent in entities_spacy:
			if ent[1] == "PERSON":
				persons.append(ent[0])

		p = ""
		for ent in entities_albert:
			if ent[1] == "B-PER":
				if p:
					persons.append(p)
				p = ent[0]
			elif ent[1] == "I-PER":
				p += " " + ent[0]
			else:
				if p:
					persons.append(p)
				p = ""
		persons = list(set([p.strip(string.punctuation) for p in persons]))
		
		return persons

	def is_actor(self, person: str) -> bool:
		""" Check if person is actor

		Keyword Arguments:
			:param person: Person to check if is actor
			:return: True if person is actor
		"""
		if "'" in person:
			person = person[:person.index("'")]
		if "`" in person:
			person = person[:person.index("`")]
		if "´" in person:
			person = person[:person.index("´")]
		
		return person.lower() in ACTORS

	def is_director(self, person: str) -> bool:
		""" Check if person is director

		Keyword Arguments:
			:param person: Person to check if is director
			:return: True if person is director
		"""
		if "'" in person:
			person = person[:person.index("'")]
		if "`" in person:
			person = person[:person.index("`")]
		if "´" in person:
			person = person[:person.index("´")]
			
		return person.lower() in DIRECTORS

	def disambiguate_person(self, person: str, text: str) -> str:
		""" Disambiguate if persos is actor or director in text

		Keyword Arguments:
			:param person: Person to disambiguate
			:param text: Text
			:return: actor | director
		"""
		if "'" in person:
			person = person[:person.index("'")]
		if "`" in person:
			person = person[:person.index("`")]
		if "´" in person:
			person = person[:person.index("´")]
			
		q_director = f"Is {person} a director?"
		q_actor = f"Is {person} an actor?"

		a_actor = qa.answer(q_actor, text)
		
		if a_actor:
			if re.findall(pat_director, a_actor, re.IGNORECASE):
				return DIRECTOR
			else:
				return ACTOR
		else:
			a_director = qa.answer(q_director, text)
			if a_director:
				return DIRECTOR
			else:
				if re.findall(pat_director, a_director, re.IGNORECASE):
					return DIRECTOR
				else:
					return ACTOR

	def get_actors_from_df(self, text: str, actors: str) -> List[str]:
		""" Get directors from df.

		Keyword Arguments:
			:param text: Text to extract actors from
			:param actors: List of actors to look for into text
			:return: List of actors extracted
		"""
		actors = actors.split(",")
		pat = fr"\b(?:{'|'.join([a.strip() for a in actors])})\b"
		actors = re.findall(pat, text, re.IGNORECASE)
		
		return actors

	def get_directors_from_df(self, text: str, directors: str) -> List[str]:
		""" Get directors from df.

		Keyword Arguments:
			:param text: Text to extract directors from
			:param directors: List of directors to look for into text
			:return: List of directors extracted
		"""
		directors = directors.split(",")
		pat = fr"\b(?:{'|'.join([d.strip() for d in directors])})\b"
		directors = re.findall(pat, text, re.IGNORECASE)
		
		return directors
		
	def run(self, **kwargs: dict) -> dict:
		""" Execute extractor. Will update kwargs with the actors, directors and characters extracted """
		persons = self.get_persons(kwargs['entities_spacy'], kwargs['entities_albert'])
		actors = []
		directors = []
		characters = []
		for person in persons:
			actor_tmp = self.is_actor(person)
			director_tmp = self.is_director(person)
			if actor_tmp and not director_tmp:
				actors.append(person)
			elif not actor_tmp and director_tmp:
				directors.append(person)
			elif not actor_tmp and not director_tmp:
				characters.append(person)
			else:
				amb = self.disambiguate_person(person, kwargs['text'])
				if amb == DIRECTOR:
					directors.append(person)
				else:
					actors.append(person)
		
		kwargs['actors'] = actors
		kwargs['directors'] = directors
		kwargs['characters'] = characters
		return kwargs