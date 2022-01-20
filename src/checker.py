import pandas as pd
from src.filter import Filter

class Checker:
	def __init__(self, filter: Filter, df: pd.DataFrame):
		""" Init checker

		Keyword Arguments:
			:param filter: Filter class to filter dataframe
			:param df: Dataframe to filter
		"""
		self.filter = filter
		self.df = df
		
	def run(self, **kwargs: dict) -> dict:
		""" Execute checker and update actors, directors, years, titles, genres	 """
		new_actors = []
		new_directors = []
		new_characters = []
		
		df = self.df.copy(deep=True)
		for actor in kwargs['actors']:
			actor_original = actor
			if "'" in actor:
				actor = actor[:actor.index("'")]
			if "`" in actor:
				actor = actor[:actor.index("`")]
			if "´" in actor:
				actor = actor[:actor.index("´")]
			df_tmp = df.copy(deep=True)
			df = self.filter.filter_by_actor(df, actor)
			if not len(df):
				df = self.filter.filter_by_director(df_tmp, actor)
				if len(df):
					new_directors.append(actor_original)
				else:
					df = df_tmp.copy(deep=True)
			else:
				new_actors.append(actor_original)
		
		for director in kwargs['directors']:
			director_original = director
			if "'" in director:
				director = director[:director.index("'")]
			if "`" in director:
				director = director[:director.index("`")]
			if "´" in director:
				director = director[:director.index("´")]
			df_tmp = df.copy(deep=True)
			df = self.filter.filter_by_director(df, director)
			if not len(df):
				df = self.filter.filter_by_actor(df_tmp, director)
				if len(df):
					new_actors.append(director_original)
				else:
					df = df_tmp.copy(deep=True)
			else:
				new_directors.append(director_original)
		
		new_years = []
		for year in kwargs['years']:
			year_ = year.strip(string.punctuation).strip(string.ascii_letters)
			df_tmp = self.filter.filter_by_year(df, year_)
			if not len(df_tmp):
				df = df.loc[df.year != year_]
			else:
				new_years.append(year)
				
		new_titles = []
		for title in kwargs['titles']:
			df_tmp = df.copy(deep=True)
			df = self.filter.filter_by_title(df, title)
			if not len(df):
				df = df_tmp.copy(deep=True)
			else:
				new_titles.append(title)
				
		new_genres = []
		for genre in kwargs['genres']:
			df_tmp = df.copy(deep=True)
			df = self.filter.filter_by_genre(df, genre)
			if not len(df):
				df = df_tmp.copy(deep=True)
			else:
				new_genres.append(genre)
		
		kwargs['actors'] = new_actors
		kwargs['directors'] = new_directors
		kwargs['years'] += new_years
		kwargs['titles'] = new_titles
		kwargs['genres'] = new_genres
		
		return kwargs, df[cols]