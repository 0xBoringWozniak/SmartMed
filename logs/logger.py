import logging

logging.basicConfig(filename='logs/start.log', level=logging.DEBUG)

def debug(fn):
	print('debug...')
	def wrapper(*args, **kwargs):
		logging.debug("Entering {:s}...".format(fn.__name__))
		result = fn(*args, **kwargs)
		logging.debug("Finished {:s}.".format(fn.__name__))
		return result

	return wrapper