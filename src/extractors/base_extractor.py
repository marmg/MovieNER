from abc import ABC, abstractmethod

class BaseExtractor(ABC):
	def __init__(self, df):
		self.df = df
		
	@abstractmethod
	def run(self, **kwargs):
		pass