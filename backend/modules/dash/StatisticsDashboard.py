import pylatex
import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


from .Dashboard import Dashboard


class StatisticsDashboard(Dashboard):

	def _generate_layout(self):
		# metrics inludings is checked inside method
		graph_list = [self._generate_table()]
		for graph in self.settings['graphs']:
			graph_list.append(self.graph_to_method[graph]())

		return html.Div(graph_list)

	def _generate_table(self, max_rows=10):
		df = self.pp.get_numeric_df(self.settings['data'])
		df = df.describe().reset_index()
		df = df[df['index'].isin(self.settings['metrics'])]
		df = df.rename(columns={"index": "metrics"})
		cols = df.columns
		markdown_text = '''
		Table illustrates main statistical characteristics of the sample.
		Среднее \n$\ \overline{x}=\ \\frac{\sum\limits_{i=1}^n\ x_i}{n}\ $
		Описательная статистика (точечная оценка), являющаяся мерой центральной тенденции для приближенно нормально распределенных данных.


		Стандартное отклонение \n$SD=\sqrt{\\frac{\sum_{i=1}^{n}\ {\ \left(\ x_i-\overline{x}\ \\right)\ }^2}{n-1}}$
		Стандартное отклонение показывает, как распределены значения относительно среднего в нашей выборке.
		Другими словами, можно понять на сколько велик разброс величины.


		Объем выборки - Число случаев, включённых в выборочную совокупность.
		От количества объектов исследования зависит мощность статистических методов, применяемых для обработки результатов эксперимента.



		Квартили – это значения признака в ранжированном ряду распределения, выбранные таким образом, что 25% единиц совокупности будут меньше по величине $Q_1$;
		25% будут заключены между $Q_1$ и $Q_2$; 25% - между $Q_2$ и $Q_3$; остальные 25% превосходят $Q_3$.
		\n$Q_1=x_{Q_1}\ +i\ \\frac{{\\frac{1}{4}}\sum f_i\ -\ S_{Q_1-1}}{f_{Q_1}}\ $
		нижняя граница интервала, содержащего нижний квартиль, i-шаг интервала,
		$𝑆_{𝑄_1−1}$-накопленные частоты интервала, предшествующего интервалу, содержащему нижний квартиль, $𝑓_{𝑄_1}$-частота интервала, содержащего нижний квартиль.


		Максимальное число (числа), не меньшее (не меньшие), чем все остальные.


		Минимальное число (числа), которое не больше всех остальных.
		'''

		for j in range(1, len(cols)):
			for i in range(len(df)):
				df.iloc[i, j] = float('{:.3f}'.format(float(df.iloc[i, j])))
		return html.Div([html.Div(html.H1(children='Desribe table'), style={'text-align': 'center'}),
						 html.Div([
							 html.Div([html.Table([
								 html.Thead(
									 html.Tr([html.Th(col)
											  for col in df.columns])
								 ),
								 html.Tbody([
									 html.Tr([
										 html.Td(df.iloc[i][col]) for col in df.columns
									 ]) for i in range(min(len(df), max_rows))
								 ]),

							 ])], style={'width': '48%', 'display': 'inline-block'}),
							 html.Div(dcc.Markdown(children=markdown_text), style={
								 'width': '48%', 'float': 'right', 'display': 'inline-block'})
						 ])
						 ], style={'margin': '50px'}
						)

	def _generate_linear(self):

		def update_graph(xaxis_column_name, yaxis_column_name,):
			fig = px.scatter(
				self.settings['data'], x=xaxis_column_name, y=yaxis_column_name)
			fig.update_xaxes(title=xaxis_column_name,
							 type='linear')
			fig.update_yaxes(title=yaxis_column_name,
							 type='linear')

			return fig

		self.app.callback(dash.dependencies.Output('linear_graph', 'figure'),
						  [dash.dependencies.Input('xaxis_column_name', 'value'),
						   dash.dependencies.Input('yaxis_column_name', 'value')])(update_graph)

		markdown_text = '''
		Каждый элемент данных представлен в виде точки на графике, заданной по двум столбцам (x и y).
		'''

		df = self.pp.get_numeric_df(self.settings['data'])
		available_indicators = df.columns.unique()

		return html.Div([html.Div(html.H1(children='Graph'), style={'text-align': 'center'}),
						 dcc.Markdown(children=markdown_text),
						 html.Div([
							 html.Div([
								 dcc.Dropdown(
									 id='xaxis_column_name',
									 options=[{'label': i, 'value': i}
											  for i in available_indicators],
									 value=available_indicators[0]
								 )
							 ], style={'width': '48%', 'display': 'inline-block'}),
							 html.Div([
								 dcc.Dropdown(
									 id='yaxis_column_name',
									 options=[{'label': i, 'value': i}
											  for i in available_indicators],
									 value=available_indicators[1]
								 )
							 ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
						 ]),
						 dcc.Graph(id='linear_graph')], style={'margin': '100px'}
						)

	def _generate_scatter(self):
		df = self.pp.get_numeric_df(self.settings['data'])
		# df.rename(columns=lambda x: x[:11], inplace=True)
		fig = px.scatter_matrix(df, width=700, height=700)
		markdown_text = '''
		На диаграммах рассеяния ряд точек отображает значения по двум переменным. 
		Сила корреляции определяется по тому, насколько близко расположены друг от друга точки на графике.
		'''
		return html.Div([html.Div(html.H1(children='Scatter matrix'), style={'text-align': 'center'}),
						 dcc.Markdown(children=markdown_text),
						 dcc.Graph(
			id='scatter_matrix',
			figure=fig
		)
		], style={'margin': '100px'})

	def _generate_heatmap(self):
		df = self.settings['data']
		numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
		df = df.select_dtypes(include=numerics)

		# df.rename(columns=lambda x: x[:11], inplace=True)
		fig = px.imshow(df)
		markdown_text = '''
		При оформлении в табличном формате тепловые карты позволяют всесторонне анализировать 
		многомерные данные за счет распределения переменных по рядам и столбцам 
		и закрашивания цветом ячеек таблицы.
		'''
		return html.Div([html.Div(html.H1(children='Heat map'), style={'text-align': 'center'}),
						 dcc.Markdown(children=markdown_text),
						 dcc.Graph(
			id='heatmap',
			figure=fig
		)
		], style={'margin': '100px'})

	def _generate_corr(self, max_rows=10):
		df = self.pp.get_numeric_df(self.settings['data'])
		# df.rename(columns=lambda x: x[:11], inplace=True)
		df = df.corr()
		cols = df.columns
		for j in range(len(cols)):
			for i in range(len(df)):
				df.iloc[i, j] = float('{:.3f}'.format(float(df.iloc[i, j])))
		markdown_text = '''
		Рассчитывает коэффициенты корреляции между всеми выбранными переменными попарно. 
		Корреляция – это мера связи между двумя переменными. Коэффициент корреляции может 
		изменяться от -1.00 до +1.00. Значение -1.00 означает полностью отрицательную корреляцию, 
		значение +1.00 означает полностью положительную корреляцию. Значение 0.00 означает отсутствие корреляции. 
		'''
		return html.Div([html.Div(html.H1(children='Correlation'), style={'text-align': 'center'}),
						 html.Div([
							 html.Div([html.Table([
								 html.Thead(
									 html.Tr([html.Th(col)
											  for col in df.columns])
								 ),
								 html.Tbody([
									 html.Tr([
										 html.Td(df.iloc[i][col]) for col in df.columns
									 ]) for i in range(min(len(df), max_rows))
								 ])
							 ])], style={'width': '48%', 'display': 'inline-block'}),
							 html.Div(dcc.Markdown(children=markdown_text), style={
								 'width': '48%', 'float': 'right', 'display': 'inline-block'})
						 ])
						 ], style={'margin': '100px'})

	def _generate_box(self):
		df = self.pp.get_numeric_df(self.settings['data'])
		# df.rename(columns=lambda x: x[:11], inplace=True)
		fig = px.box(df)
		markdown_text = '''
		Данный вид диаграммы используется для визуального представления групп числовых данных через квартили.
		Прямые линии, которые исходят из ящика, называются «усами» и используются для обозначения степени разброса 
		за пределами верхнего и нижнего квартилей
		'''
		return html.Div([html.Div(html.H1(children='Box plot'), style={'text-align': 'center'}),
						 dcc.Markdown(children=markdown_text),
						 dcc.Graph(
			id='box',
			figure=fig
		)
		], style={'margin': '100px'})

	def _generate_hist(self):
		# df.rename(columns=lambda x: x[:11], inplace=True)
		df = self.pp.get_numeric_df(self.settings['data'])
		fig = px.histogram(df)
		markdown_text = '''
		Диаграмма, построенная в столбчатой форме, в которой величина показателя изображается графически в виде столбика. 
		'''

		def update_hist(xaxis_column_name_hist):
			fig = px.histogram(
				self.settings['data'], x=xaxis_column_name_hist)
			fig.update_xaxes(title=xaxis_column_name_hist)

			return fig

		self.app.callback(dash.dependencies.Output('Histogram', 'figure'),
						  dash.dependencies.Input('xaxis_column_name_hist', 'value'))(update_hist)


		available_indicators = self.settings['data'].columns.unique()
		return html.Div([html.Div(html.H1(children='Histogram'), style={'text-align': 'center'}),
						 dcc.Markdown(children=markdown_text),

						 html.Div([
							 dcc.Dropdown(
								 id='xaxis_column_name_hist',
								 options=[{'label': i, 'value': i}
										  for i in available_indicators],
								 value=available_indicators[0]
							 )
						 ]),
						 dcc.Graph(id='Histogram')], style = {'margin': '100px'},
						)

	def _generate_box_hist(self):
		df = self.pp.get_numeric_df(self.settings['data'])
		fig_hist = px.histogram(df)
		fig_box = px.box(df)
		markdown_text = '''
		Гистограмма - это диаграмма, построенная в столбчатой форме, в которой величина показателя изображается графически в виде столбика. 
		В свою очередь, ящик с усами используется для визуального представления групп числовых данных через квартили. 
		Прямые линии, которые исходят из ящика, называются «усами» и используются для обозначения степени разброса за пределами верхнего и нижнего квартилей. 
		'''

		def update_hist(xaxis_column_name_box_hist):
			fig_hist = px.histogram(
				self.settings['data'], x=xaxis_column_name_box_hist)
			fig_hist.update_xaxes(title=xaxis_column_name_box_hist)

			return fig_hist

		self.app.callback(dash.dependencies.Output('Histogram_boxhist', 'figure'),
						  dash.dependencies.Input('xaxis_column_name_box_hist', 'value'))(update_hist)

		def update_box(xaxis_column_name_box_hist):
			fig_box = px.box(
				self.settings['data'], x=xaxis_column_name_box_hist)
			fig_box.update_xaxes(title=xaxis_column_name_box_hist)

			return fig_box

		self.app.callback(dash.dependencies.Output('Box_boxhist', 'figure'),
						  dash.dependencies.Input('xaxis_column_name_box_hist', 'value'))(update_box)

		available_indicators = self.settings['data'].columns.unique()
		return html.Div([html.Div(html.H1(children='Histogram'), style={'text-align': 'center'}),
						 dcc.Markdown(children=markdown_text),

						 html.Div([
							 dcc.Dropdown(
								 id='xaxis_column_name_box_hist',
								 options=[{'label': i, 'value': i}
										  for i in available_indicators],
								 value=available_indicators[0]
							 )
						 ]),
						 dcc.Graph(id='Histogram_boxhist'),
						 dcc.Graph(id='Box_boxhist')], style = {'margin': '100px'},
						)

	def _generate_log(self):
		markdown_text = '''	
		Каждый элемент данных представлен в виде точки на графике, заданной по двум столбцам (x и y).
		'''
		def update_graph(xaxis_column_name_log, yaxis_column_name_log,):
			fig = px.scatter(
				self.settings['data'], x=xaxis_column_name_log, y=yaxis_column_name_log)
			fig.update_xaxes(title=xaxis_column_name_log,
							 type='log')
			fig.update_yaxes(title=yaxis_column_name_log,
							 type='log')

			return fig

		self.app.callback(dash.dependencies.Output('log_graph', 'figure'),
						  [dash.dependencies.Input('xaxis_column_name_log', 'value'),
						   dash.dependencies.Input('yaxis_column_name_log', 'value')])(update_graph)

		df = self.pp.get_numeric_df(self.settings['data'])
		available_indicators = df.columns.unique()

		return html.Div([
			html.Div([
						dcc.Markdown(children=markdown_text),
						html.Div([
							dcc.Dropdown(
								id='xaxis_column_name_log',
								options=[{'label': i, 'value': i}
										 for i in available_indicators],
								value=available_indicators[0]
							)
						], style={'width': '48%', 'display': 'inline-block'}),
						html.Div([
							dcc.Dropdown(
								id='yaxis_column_name_log',
								options=[{'label': i, 'value': i}
										 for i in available_indicators],
								value=available_indicators[0]
							)
						], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
						]),
			dcc.Graph(id='log_graph')]
		)

	def _generate_linlog(self):
		markdown_text = '''	
		Каждый элемент данных представлен в виде точки на графике, заданной по двум столбцам (x и y).
		'''
		def update_graph(xaxis_column_name_linlog, yaxis_column_name_linlog,
						 xaxis_type_linlog, yaxis_type_linlog):
			fig = px.scatter(
				self.settings['data'], x=xaxis_column_name_linlog, y=yaxis_column_name_linlog)
			fig.update_xaxes(title=xaxis_column_name_linlog,
							 type='linear' if xaxis_type_linlog == 'Linear' else 'log')
			fig.update_yaxes(title=yaxis_column_name_linlog,
							 type='linear' if yaxis_type_linlog == 'Linear' else 'log')

			return fig

		self.app.callback(dash.dependencies.Output('linlog_graph', 'figure'),
						  [dash.dependencies.Input('xaxis_column_name_linlog', 'value'),
						   dash.dependencies.Input('yaxis_column_name_linlog', 'value')],
						  dash.dependencies.Input(
			'xaxis_type_linlog', 'value'),
			dash.dependencies.Input('yaxis_type_linlog', 'value'))(update_graph)

		df = self.pp.get_numeric_df(self.settings['data'])
		available_indicators = df.columns.unique()

		return html.Div([
			html.Div([
						dcc.Markdown(children=markdown_text),
						html.Div([
							dcc.Dropdown(
								id='xaxis_column_name_linlog',
								options=[{'label': i, 'value': i}
										 for i in available_indicators],
								value=available_indicators[0]
							),
							dcc.RadioItems(
								id='xaxis_type_linlog',
								options=[{'label': i, 'value': i}
										 for i in ['Linear', 'Log']],
								value='Linear'
							)
						], style={'width': '48%', 'display': 'inline-block'}),
						html.Div([
							dcc.Dropdown(
								id='yaxis_column_name_linlog',
								options=[{'label': i, 'value': i}
										 for i in available_indicators],
								value=available_indicators[0]
							),
							dcc.RadioItems(
								id='yaxis_type_linlog',
								options=[{'label': i, 'value': i}
										 for i in ['Linear', 'Log']],
								value='Linear'
							)
						], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
						]),
			dcc.Graph(id='linlog_graph')]

		)

	def _generate_piechart(self):
		df = self.pp.get_categorical_df(self.settings['data'])
		fig = px.pie(df)
		markdown_text = '''
		Данны вид диаграммы используется для отображения данных в виде круговой диаграммы, 
		в которой размер доли отображает размер конкретного параметра
		'''

		def update_pie(xaxis_column_name_pie):
			df_counts = df[xaxis_column_name_pie].value_counts()
			df_unique = df[xaxis_column_name_pie].unique()
			fig = px.pie(
				df, values=df_counts, names=df_unique)
			fig.update_xaxes(title=xaxis_column_name_pie)

			return fig

		self.app.callback(dash.dependencies.Output('Pie Chart', 'figure'),
						  dash.dependencies.Input('xaxis_column_name_pie', 'value'))(update_pie)

		available_indicators = df.columns.unique()
		return html.Div([html.Div(html.H1(children='Pie Chart'), style={'text-align': 'center'}),
						 dcc.Markdown(children=markdown_text),

						 html.Div([
							 dcc.Dropdown(
								 id='xaxis_column_name_pie',
								 options=[{'label': i, 'value': i}
										  for i in available_indicators],
								 value=available_indicators[0]
							 )
						 ]),
						 dcc.Graph(id='Pie Chart')], style={'margin': '100px'}
						)

	def _generate_dotplot(self):
		df = self.settings['data']
		df_num = self.pp.get_numeric_df(df)
		df_cat = self.pp.get_categorical_df(df)
		available_indicators_num = df_num.columns.unique()
		available_indicators_cat = df_cat.columns.unique()
		fig = go.Figure()

		fig.update_layout(title="Dot Plot",
						  xaxis_title="Value",
						  yaxis_title="Number")

		markdown_text_dotplot = 'This is Dot Plot.'

		def update_dot(xaxis_column_name_dotplot, yaxis_column_name_dotplot):
			fig = px.scatter(
				df,
				x=xaxis_column_name_dotplot,
				y=yaxis_column_name_dotplot,
				title=xaxis_column_name_dotplot,
				labels={"xaxis_column_name_dotplot": "yaxis_column_name_dotplot"}
			)

			return fig

		self.app.callback(dash.dependencies.Output('Dot Plot', 'figure'),
						  dash.dependencies.Input('xaxis_column_name_dotplot', 'value'),
						  dash.dependencies.Input('yaxis_column_name_dotplot', 'value'))(update_dot)


		return html.Div([html.Div(html.H1(children='Dotplot'), style={'text-align': 'center'}),
					 dcc.Markdown(children=markdown_text_dotplot),

					 html.Div([
						 dcc.Markdown(children="Выберите ось ОХ:"),
						 dcc.Dropdown(
							 id='xaxis_column_name_dotplot',
							 options=[{'label': i, 'value': i}
									  for i in available_indicators_num],
							 value=available_indicators_num[0]
						 ),
						 dcc.Markdown(children="Выберите ось ОY:"),
						 dcc.Dropdown(
							 id='yaxis_column_name_dotplot',
							 options=[{'label': i, 'value': i}
									  for i in available_indicators_cat],
							 value=available_indicators_cat[0]
						 )
					 ]),
					 dcc.Graph(id = 'Dot Plot', figure=fig)], style={'margin': '100px'}
					)