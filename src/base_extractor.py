from abc import ABC, abstractmethod

class BaseExtractor(ABC):
	def __init__(self, text, df)
		self.text = text
		self.df = df
		
	@abstractmethod
	def run(self, **kwargs)
		pass