import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from .markdown_bio import *

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import pandas as pd

from math import e

from .Dashboard import Dashboard

def round_df(df):
    cols = df.columns
    for j in range(0, len(cols)):
            for i in range(len(df)):
                if type(df.iloc[i, j]) != str and type(df.iloc[i, j]) != int:
                    num = str(df.iloc[i, j])
                    num = num.replace(']', '')
                    while not num[0].isdigit():
                        num = num[1:]
                    if df.iloc[i, j] < 1000 and df.iloc[i, j] >= 0.01:
                        point = num.find('.')
                        df.iloc[i, j] = num[:point + 3]
                    elif df.iloc[i, j] > 1000:
                        point = num.find('.')
                        num = num[:point]
                        df.iloc[i, j] = num[0] + '.' + num[1:3] + 'e' + str(len(num) - 1)
                    elif 'e' in num:
                        epos = num.find('e')
                        df.iloc[i, j] = num[0:4] + num[epos:]
                    elif df.iloc[i, j] < 0.01:
                        notnul = 2
                        while len(num) < notnul and num[notnul] == '0':
                            notnul += 1
                        if notnul == len(num):
                            df.iloc[i, j] = '0'
                        else:
                            df.iloc[i, j] = num[notnul] + '.' + num[notnul + 1:notnul + 3] + 'e-' + str(notnul - 1)
    return df

class BioequivalenceDashboard(Dashboard):

    def _generate_layout(self):
        # metrics inludings is checked inside method
        return html.Div(self.graphs_and_lists)

    def _generate_criteria(self):
        if self.settings[0].plan == 'parallel':
            if self.settings[0].check_normal == 'Kolmogorov' and self.settings[0].check_uniformity == 'F':
                data = {'Критерий':['Колмогорова-Смирнова', 'Колмогорова-Смирнова' , 'F-критерий'],
                    'Группа':['R', 'T', 'RT'],
                    'Значение критерия':[self.settings[0].kstest_r[0], self.settings[0].kstest_t[0], self.settings[0].f[0]],
                    'p-уровень':[self.settings[0].kstest_r[1], self.settings[0].kstest_t[1], self.settings[0].f[1]]}
            elif self.settings[0].check_normal == 'Kolmogorov' and self.settings[0].check_uniformity == 'Leven':
                data = {'Критерий':['Колмогорова-Смирнова', 'Колмогорова-Смирнова' , 'Левена'],
                'Группа':['R', 'T', 'RT'],
                'Значение критерия':[self.settings[0].kstest_r[0], self.settings[0].kstest_t[0], self.settings[0].levene[0]],
                'p-уровень':[self.settings[0].kstest_r[1], self.settings[0].kstest_t[1], self.settings[0].levene[1]]}
            elif self.settings[0].check_normal == 'Shapiro' and self.settings[0].check_uniformity == 'Leven':
                data = {'Критерий':['Шапиро-Уилка', 'Шапиро-Уилка', 'Левена'],
                'Группа':['R', 'T', 'RT'],
                'Значение критерия':[self.settings[0].shapiro_r[0], self.settings[0].shapiro_t[0], self.settings[0].levene[0]],
                'p-уровень':[self.settings[0].shapiro_r[1], self.settings[0].shapiro_t[1], self.settings[0].levene[1]]}
            else:
                data = {'Критерий':['Шапиро-Уилка', 'Шапиро-Уилка', 'F-критерий'],
                'Группа':['R', 'T', 'RT'],
                'Значение критерия':[self.settings[0].shapiro_r[0], self.settings[0].shapiro_t[0], self.settings[0].f[0]],
                'p-уровень':[self.settings[0].shapiro_r[1], self.settings[0].shapiro_t[1], self.settings[0].f[1]]}

            df = pd.DataFrame(data)
            df = round_df(df)
            return html.Div([html.Div(html.H1(children='Выполнение критериев'), style={'text-align':'center'}),
                html.Div([
                html.Div([
                    html.Div([dash_table.DataTable(
                        id='criteria',
                        columns=[{"name": i, "id": i, "deletable":True} for i in df.columns],
                        data=df.to_dict('records'),
                        style_cell_conditional=[
                        {'if': {'column_id': 'Критерий'},
                         'width': '25%'},
                        {'if': {'column_id': 'Группа'},
                         'width': '25%'},
                        {'if': {'column_id': 'Значение критерия'},
                         'width': '25%'},
                        {'if': {'column_id': 'p-уровень'},
                         'width': '25%'},
                    ]
                    )],style={'border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px','margin':'5px'})],
                    style={'width': '78%', 'display': 'inline-block'}),
                    html.Div(dcc.Markdown(children=markdown_text_criteria), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})
                    ])
                ], style={'margin':'50px'}
            )
        else:
            if self.settings[0].check_normal =='Kolmogorov':
                data = {'Критерий':['Бартлетта', 'Бартлетта', 'Колмогорова-Смирнова', 'Колмогорова-Смирнова', 'Колмогорова-Смирнова',
                    'Колмогорова-Смирнова'],
                    'Выборки':['Первая и вторая группы', 'Период 1 и период 2', 'Первая группа тестовый препарат', 'Первая группа референсный препарат', 
                    'Вторая группа тестовый препарат', 'Вторя группа референсный препарат'],
                    'Значение критерия':[self.settings[0].bartlett_groups[0], self.settings[0].bartlett_period[0], self.settings[0].kstest_t_1[0],
                    self.settings[0].kstest_r_1[0], self.settings[0].kstest_t_2[0], self.settings[0].kstest_r_2[0]],
                    'p-уровень':[self.settings[0].bartlett_groups[1], self.settings[0].bartlett_period[1], self.settings[0].kstest_t_1[1],
                    self.settings[0].kstest_r_1[1], self.settings[0].kstest_t_2[1], self.settings[0].kstest_r_2[1]]}
            else:
                data = {'Критерий':['Бартлетта', 'Бартлетта', 'Шапиро-Уилка', 'Шапиро-Уилка', 'Шапиро-Уилка',
                    'Шапиро-Уилка'],
                    'Выборки':['Первая и вторая группы', 'Период 1 и период 2', 'Первая группа тестовый препарат', 'Первая группа референсный препарат', 
                    'Вторая группа тестовый препарат', 'Вторя группа референсный препарат'],
                    'Значение критерия':[self.settings[0].bartlett_groups[0], self.settings[0].bartlett_period[0], self.settings[0].shapiro_t_1[0],
                    self.settings[0].shapiro_r_1[0], self.settings[0].shapiro_t_2[0], self.settings[0].shapiro_r_2[0]],
                    'p-уровень':[self.settings[0].bartlett_groups[1], self.settings[0].bartlett_period[1], self.settings[0].shapiro_t_1[1],
                    self.settings[0].shapiro_r_1[1], self.settings[0].shapiro_t_2[1], self.settings[0].shapiro_r_2[1]]}
            df = pd.DataFrame(data)
            df = round_df(df)
            return html.Div([html.Div(html.H1(children='Выполнение критериев'), style={'text-align':'center'}),
                html.Div([
                html.Div([
                    html.Div([dash_table.DataTable(
                        id='criteria',
                        columns=[{"name": i, "id": i, "deletable":True} for i in df.columns],
                        data=df.to_dict('records'),
                        style_cell_conditional=[
                        {'if': {'column_id': 'Критерий'},
                         'width': '25%'},
                        {'if': {'column_id': 'Выборки'},
                         'width': '25%'},
                        {'if': {'column_id': 'Значение критерия'},
                         'width': '25%'},
                        {'if': {'column_id': 'p-уровень'},
                         'width': '25%'},
                    ]
                    )],style={'border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px','margin':'5px'})],
                    style={'width': '78%', 'display': 'inline-block'}),
                    html.Div(dcc.Markdown(children=markdown_text_criteria), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})
                    ])
                ], style={'margin':'50px'}
            )

    def _generate_param(self):
        data = {'Группа':['R', 'T'],
        'AUC':[float(np.mean(self.settings[0].auc_r_notlog)), float(np.mean(self.settings[0].auc_t_notlog))],
            'AUC_inf':[float(np.mean(self.settings[0].auc_r_infty)), float(np.mean(self.settings[0].auc_t_infty))],
            'ln AUC':[float(np.mean(self.settings[0].auc_r)), float(np.mean(self.settings[0].auc_t))],
            'ln AUC_inf':[float(np.mean(self.settings[0].auc_r_infty_log)), float(np.mean(self.settings[0].auc_t_infty_log))],
            'ln Tmax':[float(np.log(self.settings[0].concentration_r.columns.max())), float(np.log(self.settings[0].concentration_t.columns.max()))],
            'ln Cmax':[float(np.log(self.settings[0].concentration_r.max().max())), float(np.log(self.settings[0].concentration_t.max().max()))]}
        df = pd.DataFrame(data)
        df = round_df(df)
        return html.Div([html.Div(html.H1(children='Распределение ключевых параметров по группам'), style={'text-align':'center'}),
            html.Div([
            html.Div([
                html.Div([dash_table.DataTable(
                    id='param',
                    columns=[{"name": i, "id": i, "deletable":True} for i in df.columns],
                    data=df.to_dict('records')
                )],style={'border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px','margin':'5px'})],
                style={'width': '78%', 'display': 'inline-block'}),
                html.Div(dcc.Markdown(children=markdown_text_param), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})
                ])
            ], style={'margin':'50px'}
        )

    def _generate_log_auc(self):
        data = {'Группа':['TR', 'RT'],
            'ln AUC T':[float(np.mean(self.settings[0].auc_t_1)), float(np.mean(self.settings[0].auc_t_2))],
            'ln AUC R':[float(np.mean(self.settings[0].auc_r_1)), float(np.mean(self.settings[0].auc_r_2))]}
        df = pd.DataFrame(data)
        df = round_df(df)
        return html.Div([html.Div(html.H1(children='Средние AUC'), style={'text-align':'center'}),
            html.Div([
            html.Div([
                html.Div([dash_table.DataTable(
                    id='param',
                    columns=[{"name": i, "id": i, "deletable":True} for i in df.columns],
                    data=df.to_dict('records')
                )],style={'border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px','margin':'5px'})],
                style={'width': '78%', 'display': 'inline-block'}),
                html.Div(dcc.Markdown(children=markdown_text_log_auc), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})
                ])
            ], style={'margin':'50px'}
        )

    def _generate_anova(self):
        if self.settings[0].plan == 'parallel':
            df = self.settings[0].anova[0]
            mark = markdown_text_anova
        else:
            df = self.settings[0].anova
            mark = markdown_text_anova_cross
        df = round_df(df)
        return html.Div([html.Div(html.H1(children='ANOVA'), style={'text-align':'center'}),
            html.Div([
            html.Div([
                html.Div([dash_table.DataTable(
                    id='anova',
                    columns=[{"name": i, "id": i, "deletable":True} for i in df.columns],
                    data=df.to_dict('records'),
                    style_cell_conditional=[
                    {'if': {'column_id': 'SS'},
                     'width': '20%'},
                    {'if': {'column_id': 'df'},
                     'width': '20%'},
                    {'if': {'column_id': 'MS'},
                     'width': '20%'},
                    {'if': {'column_id': 'F'},
                     'width': '20%'},
                    {'if': {'column_id': 'F крит.'},
                     'width': '20%'}
                ]
                )],style={'border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px','margin':'5px'})],
                style={'width': '78%', 'display': 'inline-block'}),
                html.Div(dcc.Markdown(children=mark), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})
                ])
            ], style={'margin':'50px'}
        )

    def _generate_interval(self):
        data = {'Критерий':['Биоэквивалентности', 'Бионеэквивалентности'],
            'Нижняя граница':[100*(e**self.settings[0].oneside_eq[0]), 100*(e**self.settings[0].oneside_noteq[0])],
            'Верхняя граница':[100*(e**self.settings[0].oneside_eq[1]), 100*(e**self.settings[0].oneside_noteq[1])],
            'Доверительный интервал критерия':['80.00-125.00%', '80.00-125.00%'],
            'Выполнение критерия':['Выполнен' if (self.settings[0].oneside_eq[0]>-0.223 and
            self.settings[0].oneside_eq[1]<0.223) else  'Не выполнен',
            'Выполнен' if (self.settings[0].oneside_noteq[0]>0.223 or
            self.settings[0].oneside_noteq[1]<-0.223) else  'Не выполнен']}
        df = pd.DataFrame(data)
        df = round_df(df)
        return html.Div([html.Div(html.H1(children='Результаты оценки биоэквивалентности'), style={'text-align':'center'}),
            html.Div([
            html.Div([
                html.Div([dash_table.DataTable(
                    id='interval',
                    columns=[{"name": i, "id": i} for i in df.columns],
                    data=df.to_dict('records')
                )],style={'border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px','margin':'5px'})],
                style={'width': '78%', 'display': 'inline-block'}),
                html.Div(dcc.Markdown(children=markdown_text_interval), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})
                ])
            ], style={'margin':'50px'}
        )

    def _generate_concentration_time(self, ref=True):
        if ref:
            df = self.settings[0].concentration_r
            time = df.columns

            def update_graph(yaxis_column_name_conc_r):
                fig = px.scatter(x=time, y=df.loc[yaxis_column_name_conc_r])
                fig.update_xaxes(title='Время')
                fig.update_yaxes(title=yaxis_column_name_conc_r,
                                 type='linear')
                return fig

            self.app.callback(dash.dependencies.Output('concentration_time_r', 'figure'),
                              [dash.dependencies.Input('yaxis_column_name_conc_r', 'value')])(update_graph)

            available_indicators = df.index

            return html.Div([html.Div(html.H1(children='Концентрация от времени референтный препарат'), style={'text-align':'center'}),
                html.Div([
                html.Div([
                    html.Div([
                        dcc.Markdown(children="Выберите показатель для оси ОY:"),
                        dcc.Dropdown(
                            id='yaxis_column_name_conc_r',
                            options=[{'label': i, 'value': i}
                                     for i in available_indicators],
                            value=available_indicators[0]
                        )
                    ], style={'width': '48%', 'display': 'inline-block'}),
                ], style={'padding': '5px'}),
                dcc.Graph(id='concentration_time_r')], style={'width': '78%', 'display': 'inline-block','border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
                html.Div(dcc.Markdown(children=markdown_text_conc_time_r), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
            )
        else:
            df = self.settings[0].concentration_t
            time = df.columns

            def update_graph(yaxis_column_name_conc_t):
                fig = px.scatter(x=time, y=df.loc[yaxis_column_name_conc_t])
                fig.update_xaxes(title='Время')
                fig.update_yaxes(title=yaxis_column_name_conc_t,
                                 type='linear')
                return fig

            self.app.callback(dash.dependencies.Output('concentration_time_t', 'figure'),
                              [dash.dependencies.Input('yaxis_column_name_conc_t', 'value')])(update_graph)


            available_indicators = df.index

            return html.Div([html.Div(html.H1(children='Концентрация от времени тестовый препарат'), style={'text-align':'center'}),
                html.Div([
                html.Div([
                    html.Div([
                        dcc.Markdown(children="Выберите показатель для оси ОY:"),
                        dcc.Dropdown(
                            id='yaxis_column_name_conc_t',
                            options=[{'label': i, 'value': i}
                                     for i in available_indicators],
                            value=available_indicators[0]
                        )
                    ], style={'width': '48%', 'display': 'inline-block'}),
                ], style={'padding': '5px'}),
                dcc.Graph(id='concentration_time_t')], style={'width': '78%', 'display': 'inline-block','border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
                html.Div(dcc.Markdown(children=markdown_text_conc_time_t), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
            )

    def _generate_concentration_time_log(self, ref=True):
        if ref:
            df = self.settings[0].concentration_r
            time = df.columns

            def update_graph(yaxis_column_name_conc_r_log):
                fig = px.scatter(x=time, y=df.loc[yaxis_column_name_conc_r_log])
                fig.update_xaxes(title='Время')
                fig.update_yaxes(title=yaxis_column_name_conc_r_log,
                                 type='log')
                return fig

            self.app.callback(dash.dependencies.Output('concentration_time_r_log', 'figure'),
                              [dash.dependencies.Input('yaxis_column_name_conc_r_log', 'value')])(update_graph)


            available_indicators = df.index

            return html.Div([html.Div(html.H1(children='Прологарифмированная концентрация от времени референтный препарат'), style={'text-align':'center'}),
                html.Div([
                html.Div([
                    html.Div([
                        dcc.Markdown(children="Выберите показатель для оси ОY:"),
                        dcc.Dropdown(
                            id='yaxis_column_name_conc_r_log',
                            options=[{'label': i, 'value': i}
                                     for i in available_indicators],
                            value=available_indicators[0]
                        )
                    ], style={'width': '48%', 'display': 'inline-block'}),
                ], style={'padding': '5px'}),
                dcc.Graph(id='concentration_time_r_log')], style={'width': '78%', 'display': 'inline-block','border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
                html.Div(dcc.Markdown(children=markdown_text_conc_time_r_log), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
            )
        else:
            df = self.settings[0].concentration_t
            time = df.columns

            def update_graph(yaxis_column_name_conc_t_log):
                fig = px.scatter(x=time, y=df.loc[yaxis_column_name_conc_t_log])
                fig.update_xaxes(title='Время')
                fig.update_yaxes(title=yaxis_column_name_conc_t_log,
                                 type='log')
                return fig

            self.app.callback(dash.dependencies.Output('concentration_time_t_log', 'figure'),
                              [dash.dependencies.Input('yaxis_column_name_conc_t_log', 'value')])(update_graph)


            available_indicators = df.index

            return html.Div([html.Div(html.H1(children='Прологарифмированная концентрация от времени тестовый препарат'), style={'text-align':'center'}),
                html.Div([
                html.Div([
                    html.Div([
                        dcc.Markdown(children="Выберите показатель для оси ОY:"),
                        dcc.Dropdown(
                            id='yaxis_column_name_conc_t_log',
                            options=[{'label': i, 'value': i}
                                     for i in available_indicators],
                            value=available_indicators[0]
                        )
                    ], style={'width': '48%', 'display': 'inline-block'}),
                ], style={'padding': '5px'}),
                dcc.Graph(id='concentration_time_t_log')], style={'width': '78%', 'display': 'inline-block','border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
                html.Div(dcc.Markdown(children=markdown_text_conc_time_t_log), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
            )

    def _generate_concentration_time_cross(self, tr=True):
        if tr:
            df_t = self.settings[0].concentration_t_1
            df_r = self.settings[0].concentration_r_1
            time = df_t.columns

            def update_graph(yaxis_column_name_conc_tr):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x = time, y = df_t.loc[yaxis_column_name_conc_tr], name = 'T'))
                fig.add_trace(go.Scatter(x = time, y = df_r.loc[yaxis_column_name_conc_tr], name = 'R'))
                fig.update_xaxes(title='Время')
                fig.update_yaxes(title=yaxis_column_name_conc_tr,
                                 type='linear')
                return fig

            self.app.callback(dash.dependencies.Output('concentration_time_tr', 'figure'),
                              [dash.dependencies.Input('yaxis_column_name_conc_tr', 'value')])(update_graph)

            available_indicators = df_t.index

            return html.Div([html.Div(html.H1(children='Концентрация от времени группа TR'), style={'text-align':'center'}),
                html.Div([
                html.Div([
                    html.Div([
                        dcc.Markdown(children="Выберите показатель для оси ОY:"),
                        dcc.Dropdown(
                            id='yaxis_column_name_conc_tr',
                            options=[{'label': i, 'value': i}
                                     for i in available_indicators],
                            value=available_indicators[0]
                        )
                    ], style={'width': '48%', 'display': 'inline-block'}),
                ], style={'padding': '5px'}),
                dcc.Graph(id='concentration_time_tr')], style={'width': '78%', 'display': 'inline-block','border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
                html.Div(dcc.Markdown(children=markdown_concentration_time_cross_tr), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
            )
        else:
            df_t = self.settings[0].concentration_t_2
            df_r = self.settings[0].concentration_r_2
            time = df_t.columns

            def update_graph(yaxis_column_name_conc_rt):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x = time, y = df_t.loc[yaxis_column_name_conc_rt], name = 'T'))
                fig.add_trace(go.Scatter(x = time, y = df_r.loc[yaxis_column_name_conc_rt], name = 'R'))
                fig.update_xaxes(title='Время')
                fig.update_yaxes(title=yaxis_column_name_conc_rt,
                                 type='linear')
                return fig

            self.app.callback(dash.dependencies.Output('concentration_time_rt', 'figure'),
                              [dash.dependencies.Input('yaxis_column_name_conc_rt', 'value')])(update_graph)

            available_indicators = df_t.index

            return html.Div([html.Div(html.H1(children='Концентрация от времени группа RT'), style={'text-align':'center'}),
                html.Div([
                html.Div([
                    html.Div([
                        dcc.Markdown(children="Выберите показатель для оси ОY:"),
                        dcc.Dropdown(
                            id='yaxis_column_name_conc_rt',
                            options=[{'label': i, 'value': i}
                                     for i in available_indicators],
                            value=available_indicators[0]
                        )
                    ], style={'width': '48%', 'display': 'inline-block'}),
                ], style={'padding': '5px'}),
                dcc.Graph(id='concentration_time_rt')], style={'width': '78%', 'display': 'inline-block','border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
                html.Div(dcc.Markdown(children=markdown_concentration_time_cross_rt), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
            )

    def _generate_concentration_time_cross_log(self, tr=True):
        if tr:
            df_t = self.settings[0].concentration_t_1
            df_r = self.settings[0].concentration_r_1
            time = df_t.columns

            def update_graph(yaxis_column_name_conc_tr_log):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x = time, y = df_t.loc[yaxis_column_name_conc_tr_log], name = 'T'))
                fig.add_trace(go.Scatter(x = time, y = df_r.loc[yaxis_column_name_conc_tr_log], name = 'R'))
                fig.update_xaxes(title='Время')
                fig.update_yaxes(title=yaxis_column_name_conc_tr_log,
                                 type='log')
                return fig

            self.app.callback(dash.dependencies.Output('concentration_time_tr_log', 'figure'),
                              [dash.dependencies.Input('yaxis_column_name_conc_tr_log', 'value')])(update_graph)

            available_indicators = df_t.index

            return html.Div([html.Div(html.H1(children='Логарифмированная концентрация от времени группа TR'), style={'text-align':'center'}),
                html.Div([
                html.Div([
                    html.Div([
                        dcc.Markdown(children="Выберите показатель для оси ОY:"),
                        dcc.Dropdown(
                            id='yaxis_column_name_conc_tr_log',
                            options=[{'label': i, 'value': i}
                                     for i in available_indicators],
                            value=available_indicators[0]
                        )
                    ], style={'width': '48%', 'display': 'inline-block'}),
                ], style={'padding': '5px'}),
                dcc.Graph(id='concentration_time_tr_log')], style={'width': '78%', 'display': 'inline-block','border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
                html.Div(dcc.Markdown(children=markdown_concentration_time_cross_tr_log), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
            )
        else:
            df_t = self.settings[0].concentration_t_2
            df_r = self.settings[0].concentration_r_2
            time = df_t.columns

            def update_graph(yaxis_column_name_conc_rt_log):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x = time, y = df_t.loc[yaxis_column_name_conc_rt_log], name = 'T'))
                fig.add_trace(go.Scatter(x = time, y = df_r.loc[yaxis_column_name_conc_rt_log], name = 'R'))
                fig.update_xaxes(title='Время')
                fig.update_yaxes(title=yaxis_column_name_conc_rt_log,
                                 type='log')
                return fig

            self.app.callback(dash.dependencies.Output('concentration_time_rt_log', 'figure'),
                              [dash.dependencies.Input('yaxis_column_name_conc_rt_log', 'value')])(update_graph)

            available_indicators = df_t.index

            return html.Div([html.Div(html.H1(children='Логарифмированная концентрация от времени группа RT'), style={'text-align':'center'}),
                html.Div([
                html.Div([
                    html.Div([
                        dcc.Markdown(children="Выберите показатель для оси ОY:"),
                        dcc.Dropdown(
                            id='yaxis_column_name_conc_rt_log',
                            options=[{'label': i, 'value': i}
                                     for i in available_indicators],
                            value=available_indicators[0]
                        )
                    ], style={'width': '48%', 'display': 'inline-block'}),
                ], style={'padding': '5px'}),
                dcc.Graph(id='concentration_time_rt_log')], style={'width': '78%', 'display': 'inline-block','border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
                html.Div(dcc.Markdown(children=markdown_concentration_time_cross_rt_log), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
            )

    def _generate_concentration_time_mean(self, ref=True):
        if ref:
            df = self.settings[0].concentration_r.mean()
            time = df.index

            fig = px.scatter(x = time, y = df)
            fig.update_xaxes(title='Время')
            fig.update_yaxes(type='linear')


            return html.Div([html.Div(html.H1(children='Обобщенная концентрация от времени референтный препарат'), style={'text-align':'center'}),
                html.Div([
                dcc.Graph(id='concentration_time_r_mean', figure=fig)], style={'width': '78%', 'display': 'inline-block','border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
                html.Div(dcc.Markdown(children=markdown_text_conc_time_r_mean), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
            )
        else:
            df = self.settings[0].concentration_t.mean()
            time = df.index

            fig = px.scatter(x = time, y = df)
            fig.update_xaxes(title='Время')
            fig.update_yaxes(type='linear')


            return html.Div([html.Div(html.H1(children='Обобщенная концентрация от времени тестовый препарат'), style={'text-align':'center'}),
                html.Div([
                dcc.Graph(id='concentration_time_t_mean', figure=fig)], style={'width': '78%', 'display': 'inline-block','border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
                html.Div(dcc.Markdown(children=markdown_text_conc_time_t_mean), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
            )

    def _generate_concentration_time_log_mean(self, ref=True):
        if ref:
            df = self.settings[0].concentration_r.mean()
            time = df.index

            fig = px.scatter(x = time, y = df)
            fig.update_xaxes(title='Время')
            fig.update_yaxes(type='log')


            return html.Div([html.Div(html.H1(children='Прологарифмированная обобщенная концентрация от времени референтный препарат'), style={'text-align':'center'}),
                html.Div([
                dcc.Graph(id='concentration_time_r_log_mean', figure=fig)], style={'width': '78%', 'display': 'inline-block','border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
                html.Div(dcc.Markdown(children=markdown_text_conc_time_r_log_mean), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
            )
        else:
            df = self.settings[0].concentration_t.mean()
            time = df.index

            fig = px.scatter(x = time, y = df)
            fig.update_xaxes(title='Время')
            fig.update_yaxes(type='log')


            return html.Div([html.Div(html.H1(children='Прологарифмированная обобщенная концентрация от времени тестовый препарат'), style={'text-align':'center'}),
                html.Div([
                dcc.Graph(id='concentration_time_t_log_mean', figure=fig)], style={'width': '78%', 'display': 'inline-block','border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
                html.Div(dcc.Markdown(children=markdown_text_conc_time_t_log_mean), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
            )

    def _generate_group_mean(self, tr=True):
        if tr:
            df_t = self.settings[0].concentration_t_1.mean()
            df_r = self.settings[0].concentration_r_1.mean()
            time = df_t.index

            fig = go.Figure()
            fig.add_trace(go.Scatter(x = time, y = df_t, name = 'T'))
            fig.add_trace(go.Scatter(x = time, y = df_r, name = 'R'))
            fig.update_xaxes(title='Время')
            fig.update_yaxes(type='linear')


            return html.Div([html.Div(html.H1(children='Обобщенная концентрация от времени TR'), style={'text-align':'center'}),
                html.Div([
                dcc.Graph(id='concentration_time_tr_mean', figure=fig)], style={'width': '78%', 'display': 'inline-block','border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
                html.Div(dcc.Markdown(children=markdown_group_mean_tr), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
            )
        else:
            df_t = self.settings[0].concentration_t_2.mean()
            df_r = self.settings[0].concentration_r_2.mean()
            time = df_t.index

            fig = go.Figure()
            fig.add_trace(go.Scatter(x = time, y = df_t, name = 'T'))
            fig.add_trace(go.Scatter(x = time, y = df_r, name = 'R'))
            fig.update_xaxes(title='Время')
            fig.update_yaxes(type='linear')


            return html.Div([html.Div(html.H1(children='Обобщенная концентрация от времени RT'), style={'text-align':'center'}),
                html.Div([
                dcc.Graph(id='concentration_time_rt_mean', figure=fig)], style={'width': '78%', 'display': 'inline-block','border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
                html.Div(dcc.Markdown(children=markdown_group_mean_rt), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
            )

    def _generate_group_mean_log(self, tr=True):
        if tr:
            df_t = self.settings[0].concentration_t_1.mean()
            df_r = self.settings[0].concentration_r_1.mean()
            time = df_t.index

            fig = go.Figure()
            fig.add_trace(go.Scatter(x = time, y = df_t, name = 'T'))
            fig.add_trace(go.Scatter(x = time, y = df_r, name = 'R'))
            fig.update_xaxes(title='Время')
            fig.update_yaxes(type='log')


            return html.Div([html.Div(html.H1(children='Логарифмированная обобщенная концентрация от времени TR'), style={'text-align':'center'}),
                html.Div([
                dcc.Graph(id='concentration_time_tr_mean_log', figure=fig)], style={'width': '78%', 'display': 'inline-block','border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
                html.Div(dcc.Markdown(children=markdown_group_mean_tr_log), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
            )
        else:
            df_t = self.settings[0].concentration_t_2.mean()
            df_r = self.settings[0].concentration_r_2.mean()
            time = df_t.index

            fig = go.Figure()
            fig.add_trace(go.Scatter(x = time, y = df_t, name = 'T'))
            fig.add_trace(go.Scatter(x = time, y = df_r, name = 'R'))
            fig.update_xaxes(title='Время')
            fig.update_yaxes(type='log')


            return html.Div([html.Div(html.H1(children='Логарифмированная обобщенная концентрация от времени RT'), style={'text-align':'center'}),
                html.Div([
                dcc.Graph(id='concentration_time_rt_mean_log', figure=fig)], style={'width': '78%', 'display': 'inline-block','border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
                html.Div(dcc.Markdown(children=markdown_group_mean_rt_log), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
            )


    def _generate_drug_mean(self):
        df_t_1 = self.settings[0].concentration_t_1.mean()
        df_t_2 = self.settings[0].concentration_t_2.mean()
        df_r_1 = self.settings[0].concentration_r_1.mean()
        df_r_2 = self.settings[0].concentration_r_2.mean()
        time = df_t_1.index

        fig = go.Figure()
        fig.add_trace(go.Scatter(x = time, y = (df_t_1 + df_t_2) / 2, name = 'T'))
        fig.add_trace(go.Scatter(x = time, y = (df_r_1 + df_r_2) / 2, name = 'R'))
        fig.update_xaxes(title='Время')
        fig.update_yaxes(type='linear')


        return html.Div([html.Div(html.H1(children='Обобщенная концентрация тестовый препарат'), style={'text-align':'center'}),
            html.Div([
            dcc.Graph(id='drug_mean', figure=fig)], style={'width': '78%', 'display': 'inline-block','border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
            html.Div(dcc.Markdown(children=markdown_drug_mean), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
        )

    def _generate_drug_mean_log(self):
        df_t_1 = self.settings[0].concentration_t_1.mean()
        df_t_2 = self.settings[0].concentration_t_2.mean()
        df_r_1 = self.settings[0].concentration_r_1.mean()
        df_r_2 = self.settings[0].concentration_r_2.mean()
        time = df_t_1.index

        fig = go.Figure()
        fig.add_trace(go.Scatter(x = time, y = (df_t_1 + df_t_2) / 2, name = 'T'))
        fig.add_trace(go.Scatter(x = time, y = (df_r_1 + df_r_2) / 2, name = 'R'))
        fig.update_xaxes(title='Время')
        fig.update_yaxes(type='log')


        return html.Div([html.Div(html.H1(children='Обобщенная концентрация тестовый препарат'), style={'text-align':'center'}),
            html.Div([
            dcc.Graph(id='drug_mean_log', figure=fig)], style={'width': '78%', 'display': 'inline-block','border-color': 'rgb(220, 220, 220)','border-style': 'solid','padding':'5px'}),
            html.Div(dcc.Markdown(children=markdown_drug_mean_log), style={'width': '18%', 'float': 'right', 'display': 'inline-block'})], style={'margin':'100px'}
        )