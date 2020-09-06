from .apps import *

# logging decorator
import sys
sys.path.append("..")
from logs.logger import debug


class GUI:
	'''Qt apps manipulator'''
	def __init__(self):
		pass

	@debug
	def start_gui(self):
		pass
