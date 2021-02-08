import pylatex
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from .markdown_text import *

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
		len_t = str(len(df.columns)*10-10)+'%'
		len_text = str(98-len(df.columns)*10+10)+'%'
		for j in range(1,len(cols)):
			for i in range(len(df)):
				df.iloc[i, j] = float('{:.3f}'.format(float(df.iloc[i, j])))
		if len(df.columns) <= 5:
			return html.Div([html.Div(html.H1(children='Описательная таблица'), style={'text-align':'center'}),
				html.Div([
				html.Div([
					html.Div([dash_table.DataTable(
					    id='table',
					    columns=[{"name": i, "id": i, "deletable":True} for i in df.columns],
					    data=df.to_dict('records')
					)],style={'border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px','margin':'5px'})],
					style={'width': len_t, 'display': 'inline-block'}),
					html.Div(dcc.Markdown(children=markdown_text_table), style={'width': len_text, 'float': 'right', 'display': 'inline-block'})
					])
				], style={'margin':'50px'}
			)
		else:
			return html.Div([html.Div(html.H1(children='Описательная таблица'), style={'text-align':'center'}),
				dcc.Markdown(children=markdown_text_table),
					html.Div([dash_table.DataTable(
					    id='table',
					    columns=[{"name": i, "id": i, "deletable":True} for i in df.columns],
					    data=df.to_dict('records')
    			)],style={'border-color':'rgb(220, 220, 220)','border-style': 'solid','padding':'5px','margin':'5px'})
				], style={'margin':'50px'}
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


		df = self.pp.get_numeric_df(self.settings['data'])
		available_indicators = df.columns.unique()

		return html.Div([html.Div(html.H1(children='Линейный график'), style={'text-align':'center'}),
			html.Div([
			html.Div([
				html.Div([
					dcc.Markdown(children="Выберите показатель для оси ОХ:"),
					dcc.Dropdown(
						id='xaxis_column_name',
						options=[{'label': i, 'value': i}
								 for i in available_indicators],
						value=available_indicators[0]
					)
				], style={'width': '48%', 'display': 'inline-block'}),
				html.Div([
					dcc.Markdown(children="Выберите показатель для оси ОY:"),
					dcc.Dropdown(
						id='yaxis_column_name',
						options=[{'label': i, 'value': i}
								 for i in available_indicators],
						 value=available_indicators[1]
					)
				], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
			], style={'padding': '5px'}),
			dcc.Graph(id='linear_graph')], style={'width': '78%', 'display': 'inline-block','border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
			html.Div(dcc.Markdown(children=markdown_text_lin), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
		)

	def _generate_scatter(self):
		df = self.pp.get_numeric_df(self.settings['data']).copy()
		df.rename(columns=lambda x: x[:11], inplace=True)
		fig = px.scatter_matrix(df, height=700)
		fig.update_xaxes(tickangle=90)
		for annotation in fig['layout']['annotations']: 
			annotation['textangle']=-90
		return html.Div([html.Div(html.H1(children='Матрица рассеяния'), style={'text-align':'center'}),
			html.Div([
				html.Div(dcc.Graph(
        			id='scatter_matrix',
        			figure=fig
    			),style={'width': '78%', 'display': 'inline-block','border-color':'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
    			html.Div(dcc.Markdown(children=markdown_text_scatter), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})])
			], style={'margin':'100px'})

	def _generate_heatmap(self):
		df = self.pp.get_numeric_df(self.settings['data']).copy()
		numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
		df = df.select_dtypes(include=numerics)
		df.rename(columns=lambda x: x[:11], inplace=True)
		fig = px.imshow(df)
		return html.Div([html.Div(html.H1(children='Хитмап'), style={'text-align':'center'}),
			html.Div([
				html.Div(dcc.Graph(
        			id='heatmap',
        			figure=fig
    			),style={'width': '78%', 'display': 'inline-block','border-color':'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
    			html.Div(dcc.Markdown(children=markdown_text_heatmap), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})])
			], style={'margin':'100px'})

	def _generate_corr(self, max_rows=10):
		df = self.pp.get_numeric_df(self.settings['data'])
		#df.rename(columns=lambda x: x[:11], inplace=True)

		df = df.corr()
		cols = df.columns
		len_t = str(len(df.columns)*10)+'%'
		len_text = str(98-len(df.columns)*10)+'%'
		for j in range(len(cols)):
			for i in range(len(df)):
				df.iloc[i, j] = float('{:.3f}'.format(float(df.iloc[i, j])))
		if len(df.columns) <= 5:
			return html.Div([html.Div(html.H1(children='Таблица корреляций'), style={'text-align':'center'}),
				html.Div([
				html.Div([
					html.Div([dash_table.DataTable(
					    id='corr',
					    columns=[{"name": i, "id": i} for i in df.columns],
					    data=df.to_dict('records')
					    )],style={'border-color':'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'})],
					style={'width': len_t, 'display': 'inline-block'}),
					html.Div(dcc.Markdown(children=markdown_text_corr), style={'width': len_text, 'float': 'right', 'display': 'inline-block'})
					])
				], style={'margin':'50px'}
			)
		else:
			return html.Div([html.Div(html.H1(children='Таблица корреляций'), style={'text-align':'center'}),
				dcc.Markdown(children=markdown_text_corr),
					html.Div([dash_table.DataTable(
					    id='corr',
					    columns=[{"name": i, "id": i} for i in df.columns],
					    data=df.to_dict('records'))
					],style={'border-color':'rgb(192, 192, 192)','border-style': 'solid','padding':'5px'})	
				], style={'margin':'50px'}
			)

	def _generate_box(self):
		df = self.pp.get_numeric_df(self.settings['data'])
		#df.rename(columns=lambda x: x[:11], inplace=True)
		fig = px.box(df)
		return html.Div([html.Div(html.H1(children='Ящик с усами'), style={'text-align':'center'}),
			html.Div([
				html.Div(dcc.Graph(
        			id='box',
        			figure=fig
    			),style={'width': '78%', 'display': 'inline-block','border-color':'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
    			html.Div(dcc.Markdown(children=markdown_text_box), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})])
			], style={'margin':'100px'})

	def _generate_hist(self):
		def update_hist(xaxis_column_name_hist):
			fig = go.Figure(data=go.Histogram(
				x=self.pp.get_numeric_df(self.settings['data'])[xaxis_column_name_hist]))
			fig.update_xaxes(title=xaxis_column_name_hist)
			fig.update_layout(bargap=0.1)

			return fig

		self.app.callback(dash.dependencies.Output('Histogram', 'figure'),
						  dash.dependencies.Input('xaxis_column_name_hist', 'value'))(update_hist)

		available_indicators = self.pp.get_numeric_df(self.settings['data']).columns.unique()

		return html.Div([html.Div(html.H1(children='Гистограмма'), style={'text-align': 'center'}),
						 html.Div([
							 dcc.Markdown(children="Выберите показатель:"),
							 dcc.Dropdown(
								 id='xaxis_column_name_hist',
								 options=[{'label': i, 'value': i}
										  for i in available_indicators],
								 value=available_indicators[0]
							 ),
							 dcc.Graph(id='Histogram')],
							 style={'width': '78%', 'display': 'inline-block', 'border-color': 'rgb(220, 220, 220)',
									'border-style': 'solid', 'padding': '5px'}),
						 html.Div(dcc.Markdown(children=markdown_text_hist),
								  style={'width': '18%', 'float': 'right', 'display': 'inline-block'})],
						style={'margin': '100px'}
						)

	def _generate_log(self):

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

		return html.Div([html.Div(html.H1(children='Логарифмический график'), style={'text-align':'center'}),
			html.Div([
			html.Div([
				html.Div([
					dcc.Markdown(children="Выберите показатель для оси ОХ:"),
					dcc.Dropdown(
						id='xaxis_column_name_log',
						options=[{'label': i, 'value': i}
								 for i in available_indicators],
						value=available_indicators[0]
					)
				], style={'width': '48%', 'display': 'inline-block'}),
				html.Div([
					dcc.Markdown(children="Выберите показатель для оси ОY:"),
					dcc.Dropdown(
						id='yaxis_column_name_log',
						options=[{'label': i, 'value': i}
								 for i in available_indicators],
						 value=available_indicators[1]
					)
				], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
			], style={'padding': '5px'}),
			dcc.Graph(id='log_graph')], style={'width': '78%', 'display': 'inline-block','border-color':'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
			html.Div(dcc.Markdown(children=markdown_text_log), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
		)

	def _generate_linlog(self):

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

		return html.Div([html.Div(html.H1(children='Линейный/логарифмический график'), style={'text-align':'center'}),
			html.Div([
			html.Div([
						html.Div([
							dcc.Markdown(children="Выберите показатель для оси ОХ:"),
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
							dcc.Markdown(children="Выберите показатель для оси ОY:"),
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
						], style={'padding': '5px'}),
			dcc.Graph(id='linlog_graph')], style={'width': '78%', 'display': 'inline-block','border-color':'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
			html.Div(dcc.Markdown(children=markdown_text_linlog), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}

		)

	def _generate_piechart(self):
		df = self.pp.get_categorical_df(self.settings['data'])
		fig = px.pie(df)

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

		if df.size > 0:
			return html.Div([html.Div(html.H1(children='Круговая диаграмма'), style={'text-align': 'center'}),
						 html.Div([
							 html.Div([
								 html.Div([
									 dcc.Markdown(children="Выберите показатель для оси ОX:"),
									 dcc.Dropdown(
										 id='xaxis_column_name_pie',
										 options=[{'label': i, 'value': i}
												  for i in available_indicators],
										 value=available_indicators[0]
									 )

								 ], style={'width': '48%', 'display': 'inline-block', 'padding': '5px'})
							 ]),
							 dcc.Graph(id='Pie Chart', figure=fig)], style={'width': '78%', 'display': 'inline-block',
																			'border-color': 'rgb(220, 220, 220)',
																			'border-style': 'solid', 'padding': '5px'}),
						 html.Div(dcc.Markdown(children=markdown_text_pie), style={'width': '18%', 'float': 'right',
																				   'display': 'inline-block'})],
						style={'margin': '100px'})
		else:
			return html.Div([html.Div(html.H1(children='Круговая диаграмма'), style={'text-align': 'center'}),
						 html.Div(dcc.Markdown(
							 children='Ошибка: невозможно построить круговую диаграмму, т.к. нет категориальных данных.'),
							 style={'width': '80%', 'display': 'inline-block'})],
						style={'margin': '100px'})

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


		def update_dot(xaxis_column_name_dotplot, yaxis_column_name_dotplot):
			fig = px.scatter(
				df,
				x=xaxis_column_name_dotplot,
				y=yaxis_column_name_dotplot,
				#title=xaxis_column_name_dotplot,
				labels={"xaxis_column_name_dotplot": "yaxis_column_name_dotplot"}
			)

			return fig

		self.app.callback(dash.dependencies.Output('Dot Plot', 'figure'),
							  dash.dependencies.Input('xaxis_column_name_dotplot', 'value'),
							  dash.dependencies.Input('yaxis_column_name_dotplot', 'value'))(update_dot)

		if df_cat.size > 0:
			return html.Div([html.Div(html.H1(children='Точечная диаграмма'), style={'text-align': 'center'}),
						 html.Div([
							html.Div([
								html.Div([
									dcc.Markdown(children="Выберите показатель для оси ОХ:"),
									dcc.Dropdown(
										id='xaxis_column_name_dotplot',
										options=[{'label': i, 'value': i}
												 for i in available_indicators_num],
										value=available_indicators_num[0]
									)
								], style={'width': '48%', 'display': 'inline-block'}),
								html.Div([
									dcc.Markdown(children="Выберите показатель для оси ОY:"),
									dcc.Dropdown(
										id='yaxis_column_name_dotplot',
										options=[{'label': i, 'value': i}
												 for i in available_indicators_cat],
										value=available_indicators_cat[0]
									)

								], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
							], style={'padding': '5px'}),
					     dcc.Graph(id='Dot Plot', figure=fig)], style={'width': '78%', 'display': 'inline-block',
																		'border-color': 'rgb(220, 220, 220)',
																		'border-style': 'solid', 'padding':'5px'}),
						 html.Div(dcc.Markdown(children=markdown_text_dotplot),style={'width': '18%', 'float': 'right',
															'display': 'inline-block'})],style={'margin': '100px'})
		else:
			return html.Div([html.Div(html.H1(children='Точечная диаграмма'), style={'text-align': 'center'}),
						 html.Div(dcc.Markdown(
							 children='Ошибка: невозможно построить точечную диаграмму, т.к. нет категориальных данных.'),
							 style={'width': '80%', 'display': 'inline-block'})
						], style={'margin': '100px'})

	def _generate_box_hist(self):
		df = self.pp.get_numeric_df(self.settings['data'])
		fig_hist = px.histogram(df)
		fig_box = px.box(df)
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

		return html.Div([html.Div(html.H1(children='Гистограмма и ящик с усами'), style={'text-align': 'center'}),
						 html.Div([
							html.Div([
								html.Div([
									dcc.Markdown(children="Выберите показатель:"),
									dcc.Dropdown(
										id='xaxis_column_name_box_hist',
										options=[{'label': i, 'value': i}
												 for i in available_indicators],
										value=available_indicators[0]
									)

								], style={'width': '48%', 'display': 'inline-block', 'padding': '5px'})
							]),
							 html.Div([
								 dcc.Graph(id='Histogram_boxhist', figure=fig_hist),
								 dcc.Graph(id='Box_boxhist', figure=fig_box)
							 ])
						 ], style={'width': '78%', 'display': 'inline-block',
									   'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
									   'padding': '5px'}),
						 html.Div(dcc.Markdown(children=markdown_text_histbox),style={'width': '18%', 'float': 'right',
															'display': 'inline-block'})],style={'margin': '100px'})

