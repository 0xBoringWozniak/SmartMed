import pandas as pd
import dash

from constructor import LayoutConstructor 


if __name__ == '__main__':
	settings = {'data': {
						'path': 'fr_bitmex.csv'
						}
				}

	constructor = LayoutConstructor(settings)
	constructor.start(debug=False)
