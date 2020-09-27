import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.express as px
import pandas as pd

from .Dashboard import Dashboard


class StatisticsDashboard(Dashboard):
	def _generate_layout(self):
		return html.Div([self._generate_table(),
						 self._generate_linear(),
						 self._generate_scatter(),
						 self._generate_heatmap(),
						 self._generate_corr(),
						 self._generate_linlog()])

	def _generate_table(self, max_rows=10):
		df = self.settings['data'].describe().reset_index()
		df = df[df['index'].isin(self.settings['metrics'])]
		cols = df.columns[1:]
		for col in cols:
			for i in range(len(df)):
				df.iloc[i][col] = float('{:.3f}'.format(float(df.iloc[i][col].copy())))
		return html.Table([
			html.Thead(
				html.Tr([html.Th(col) for col in df.columns])
			),
			html.Tbody([
				html.Tr([
					html.Td(df.iloc[i][col]) for col in df.columns
				]) for i in range(min(len(df), max_rows))
			])
		])

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

		available_indicators = self.settings['data'].columns.unique()
		return html.Div([
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
						value=available_indicators[0]
					)
				], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
			]),
			dcc.Graph(id='linear_graph')]
		)

	def _generate_scatter(self):
		df = self.settings['data']
		df.rename(columns=lambda x: x[:11], inplace=True)
		fig = px.scatter_matrix(df, width=700, height=700)
		return html.Div(
				dcc.Graph(
        			id='scatter_matrix',
        			figure=fig
    			)
			)

	def _generate_heatmap(self):
		df = self.settings['data']
		df.rename(columns=lambda x: x[:11], inplace=True)
		fig = px.imshow(df)
		return html.Div(
				dcc.Graph(
        			id='heatmap',
        			figure=fig
    			)
			)

	def _generate_corr(self, max_rows=10):
		df = self.settings['data']
		df.rename(columns=lambda x: x[:11], inplace=True)
		df = df.corr()
		return html.Table([
			html.Thead(
				html.Tr([html.Th(col) for col in df.columns])
			),
			html.Tbody([
				html.Tr([
					html.Td(df.iloc[i][col]) for col in df.columns
				]) for i in range(min(len(df), max_rows))
			])
		])

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

		available_indicators = self.settings['data'].columns.unique()
		return html.Div([
			html.Div([
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
						  dash.dependencies.Input('xaxis_type_linlog', 'value'),
						  dash.dependencies.Input('yaxis_type_linlog', 'value'))(update_graph)

		available_indicators = self.settings['data'].columns.unique()
		return html.Div([
			html.Div([
				html.Div([
					dcc.Dropdown(
						id='xaxis_column_name_linlog',
						options=[{'label': i, 'value': i}
								 for i in available_indicators],
						value=available_indicators[0]
					),
					dcc.RadioItems(
						id='xaxis_type_linlog',
						options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
						value='Linear'
					)
				], style={'width': '48%', 'display': 'inline-block'}),
				html.Div([
					dcc.Dropdown(
						id='yaxis_column_name_linlog',
						options=[{'label': i, 'value': i} for i in available_indicators],
						value=available_indicators[0]
					),
					dcc.RadioItems(
						id='yaxis_type_linlog',
						options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
						value='Linear'
					)
				], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
			]),
			dcc.Graph(id='linlog_graph')]
		)

