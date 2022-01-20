from .albert import AlbertQA
from .endpoint_config import ASSETS_PATH, MODELS_PATH

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
	def get_persons(self, entities_spacy, entities_albert):
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

	def is_actor(self, person):
		if "'" in person:
			person = person[:person.index("'")]
		if "`" in person:
			person = person[:person.index("`")]
		if "´" in person:
			person = person[:person.index("´")]
		
		return person.lower() in ACTORS

	def is_director(self, person):
		if "'" in person:
			person = person[:person.index("'")]
		if "`" in person:
			person = person[:person.index("`")]
		if "´" in person:
			person = person[:person.index("´")]
			
		return person.lower() in DIRECTORS

	def disambiguate_person(self, person):
		if "'" in person:
			person = person[:person.index("'")]
		if "`" in person:
			person = person[:person.index("`")]
		if "´" in person:
			person = person[:person.index("´")]
			
		q_director = f"Is {person} a director?"
		q_actor = f"Is {person} an actor?"

		a_actor = qa.answer(q_actor, self.text)
		
		if a_actor:
			if re.findall(pat_director, a_actor, re.IGNORECASE):
				return DIRECTOR
			else:
				return ACTOR
		else:
			a_director = qa.answer(q_director, self.text)
			if a_director:
				return DIRECTOR
			else:
				if re.findall(pat_director, a_director, re.IGNORECASE):
					return DIRECTOR
				else:
					return ACTOR

	def get_actors_from_df(self, actors):
		actors = actors.split(",")
		pat = fr"\b(?:{'|'.join([a.strip() for a in actors])})\b"
		actors = re.findall(pat, self.text, re.IGNORECASE)
		
		return actors

	def get_directors_from_df(self, directors):
		directors = directors.split(",")
		pat = fr"\b(?:{'|'.join([d.strip() for d in directors])})\b"
		directors = re.findall(pat, self.text, re.IGNORECASE)
		
		return directors
		
	def run(self, **kwargs):
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
				amb = self.disambiguate_person(person, text)
				if amb == DIRECTOR:
					directors.append(person)
				else:
					actors.append(person)
		
		kwargs['actors'] = actors
		kwargs['directors'] = directors
		return kwargs