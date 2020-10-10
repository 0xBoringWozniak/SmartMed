from GUI import GUI
from backend import ModuleManipulator


if __name__ == '__main__':

	try:
		settings = GUI().start_gui()
	except Exception as e:
		print('GUI ERROR: ', e)

	try:
		ModuleManipulator(settings).start()
	except Exception as e:
		print('BACKEND ERROR: ', e)

