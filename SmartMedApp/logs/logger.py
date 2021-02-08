import logging

logging.basicConfig(filename='logs/start.log', level=logging.DEBUG)


def debug(fn):
	print('debug...')

	def wrapper(*args, **kwargs):
		print("Entering {:s}.{:s}...".format(fn.__module__,
													 fn.__name__))
		result = fn(*args, **kwargs)
		print("Finished {:s}.{:s}.".format(fn.__module__,
												   fn.__name__))
		return result

	return wrapper
