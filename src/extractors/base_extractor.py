from abc import ABC, abstractmethod
import pandas as pd

class BaseExtractor(ABC):
	""" Base Extractor"""
	def __init__(self, df: pd.DataFrame):
		self.df = df
		
	@abstractmethod
	def run(self, **kwargs):
		pass