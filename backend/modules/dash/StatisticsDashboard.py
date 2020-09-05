import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd


class StatisticsDashboard():
    def __init__(self):
        self.app = dash.Dash(server=True)
        self.app.callback(dash.dependencies.Output('linear_graph', 'figure'),
                          [dash.dependencies.Input('xaxis_column_name', 'value'),
                           dash.dependencies.Input('yaxis_column_name', 'value')])(self.update_graph)


    def update_graph(self, xaxis_column_name, yaxis_column_name,):
        print(1)
        fig = px.scatter(self.settings['data'], x=xaxis_column_name, y=yaxis_column_name)
        fig.update_xaxes(title=xaxis_column_name,
                         type='linear')
        fig.update_yaxes(title=yaxis_column_name,
                         type='linear')

        return fig

    def _generate_layout(self):
        return html.Div([self._generate_table(),
                         self._generate_linear()])

    def _generate_table(self, max_rows=10):
        df = self.settings['data'].describe().reset_index()
        df = df[df['index'].isin(self.settings['metrics'])]
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
        available_indicators = self.settings['data'].columns.unique()
        return html.Div([
                        html.Div([
                            html.Div([
                                dcc.Dropdown(
                                    id='xaxis_column_name',
                                    options=[{'label': i, 'value': i}
                                             for i in available_indicators],
                                    value=available_indicators[2]
                                )
                            ], style={'width': '48%', 'display': 'inline-block'}),
                            html.Div([
                                dcc.Dropdown(
                                    id='yaxis_column_name',
                                    options=[{'label': i, 'value': i}
                                             for i in available_indicators],
                                    value=available_indicators[3]
                                )
                            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
                        ]),
                        dcc.Graph(id='linear_graph')]
                        )

    def start(self, debug=False):
        self.app.layout = self._generate_layout()
        self.app.run_server(debug=debug)
