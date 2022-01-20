class Filter:
	def __init__(self, df):
		self.df = df
		
	def filter_by_director(self, director):
		return self.df.loc[self.df.director.str.lower() == director.lower()]

	def filter_by_actor(self, actor):
		return pd.DataFrame([mov_ for mov_ in self.df.itertuples() if actor.lower() in mov_.actors.lower()])
		
	def filter_by_genre(self, genre):
		return pd.DataFrame([mov_ for mov_ in self.df.itertuples() if genre.lower() in mov_.genre.lower()])
	
	def filter_by_title(self, title):
			return self.df.loc[self.df.original_title.str.lower() == title.lower()]	
		
	def filter_by_year(self, year):
		return self.df.loc[self.df.year == year]
	