import pandas as pd


class Filter:
	""" Filter Class to filter DataFrame """
	def filter_by_director(self, df: pd.DataFrame, director: str) -> pd.DataFrame:
		""" Filter DataFrame by director

		Keyword Arguments:
			:param df: (pd.DataFrame) DataFrame to filter
			:param director: (str) Director to filter df by
			:return: (pd.DataFrame) DataFrame with those movies directed by director
		"""
		return df.loc[df.director.str.lower() == director.lower()]

	def filter_by_actor(self, df: pd.DataFrame, actor: str) -> pd.DataFrame:
		""" Filter DataFrame by actor

		Keyword Arguments:
			:param df: (pd.DataFrame) DataFrame to filter
			:param actor: (str) Actor to filter df by
			:return: (pd.DataFrame) DataFrame with those movies that actor acts in
		"""
		return pd.DataFrame([mov_ for mov_ in df.itertuples() if actor.lower() in mov_.actors.lower()])
		
	def filter_by_genre(self, df: pd.DataFrame, genre: str) -> pd.DataFrame:
		""" Filter DataFrame by genre

		Keyword Arguments:
			:param df: (pd.DataFrame) DataFrame to filter
			:param genre: (str) Genre to filter df by
			:return: (pd.DataFrame) DataFrame with those movies that belong to genre
		"""
		return pd.DataFrame([mov_ for mov_ in df.itertuples() if genre.lower() in mov_.genre.lower()])
	
	def filter_by_title(self, df: pd.DataFrame, title: str) -> pd.DataFrame:
		""" Filter DataFrame by title

		Keyword Arguments:
			:param df: (pd.DataFrame) DataFrame to filter
			:param title: (str) Title to filter df by
			:return: (pd.DataFrame) DataFrame with those movies that have title
		"""
		return df.loc[df.original_title.str.lower() == title.lower()]
		
	def filter_by_year(self, df: pd.DataFrame, year: str) -> pd.DataFrame:
		""" Filter DataFrame by year

		Keyword Arguments:
			:param df: (pd.DataFrame) DataFrame to filter
			:param year: (str) Year to filter df by
			:return: (pd.DataFrame) DataFrame with those movies released in year
		"""
		return df.loc[df.year == year]
	