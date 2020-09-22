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
						 self._generate_corr()])

	def _generate_table(self, max_rows=10):
		df = self.settings['data'].describe().reset_index()
		df = df[df['index'].isin(self.settings['metrics'])]
		cols = df.columns
		markdown_text='''
		Table illustrates main statistical characteristics of the sample.
		'''
		for j in range(1,len(cols)):
			for i in range(len(df)):
				df.iloc[i, j] = float('{:.3f}'.format(float(df.iloc[i, j])))
		return html.Div([html.Div(html.H1(children='Desribe table'), style={'text-align':'center'}),
			html.Div([
				html.Div([html.Table([
				html.Thead(
					html.Tr([html.Th(col) for col in df.columns])
				),
				html.Tbody([
					html.Tr([
						html.Td(df.iloc[i][col]) for col in df.columns
					]) for i in range(min(len(df), max_rows))
				])
			])],style={'width': '48%', 'display': 'inline-block'}),
				html.Div(dcc.Markdown(children=markdown_text), style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
				])
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

		markdown_text='''
		Graph shows relation between to columns of the data (x-axis coordinates are values of the data column chosen in the left dropdown
		and y-axis coordinate are values of the data column chosen in the right dropdown).
		'''

		available_indicators = self.settings['data'].columns.unique()
		return html.Div([html.Div(html.H1(children='Graph'), style={'text-align':'center'}),
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
			dcc.Graph(id='linear_graph')], style={'margin':'100px'}
		)

	def _generate_scatter(self):
		df = self.settings['data']
		df.rename(columns=lambda x: x[:11], inplace=True)
		fig = px.scatter_matrix(df, width=700, height=700)
		markdown_text='''
		Scatter matrix illustrates correlation between all columns of the data.
		'''
		return html.Div([html.Div(html.H1(children='Scatter matrix'), style={'text-align':'center'}),
			dcc.Markdown(children=markdown_text),
				dcc.Graph(
        			id='scatter_matrix',
        			figure=fig
    			)
			], style={'margin':'100px'})

	def _generate_heatmap(self):
		df = self.settings['data']
		df.rename(columns=lambda x: x[:11], inplace=True)
		fig = px.imshow(df)
		markdown_text='''
		Heat map depicts the magnitude for each column using different colors.
		'''
		return html.Div([html.Div(html.H1(children='Heat map'), style={'text-align':'center'}),
			dcc.Markdown(children=markdown_text),
				dcc.Graph(
        			id='heatmap',
        			figure=fig
    			)
			], style={'margin':'100px'})

	def _generate_corr(self, max_rows=10):
		df = self.settings['data']
		df.rename(columns=lambda x: x[:11], inplace=True)
		df = df.corr()
		cols = df.columns
		for j in range(len(cols)):
			for i in range(len(df)):
				df.iloc[i, j] = float('{:.3f}'.format(float(df.iloc[i, j])))
		markdown_text='''
		Table shows pairwise pearson correlation of columns.
		'''
		return html.Div([html.Div(html.H1(children='Correlation'), style={'text-align':'center'}),
			html.Div([
				html.Div([html.Table([
				html.Thead(
					html.Tr([html.Th(col) for col in df.columns])
				),
				html.Tbody([
					html.Tr([
						html.Td(df.iloc[i][col]) for col in df.columns
					]) for i in range(min(len(df), max_rows))
				])
			])], style={'width': '48%', 'display': 'inline-block'}),
				html.Div(dcc.Markdown(children=markdown_text), style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
		])
		], style={'margin':'100px'})


