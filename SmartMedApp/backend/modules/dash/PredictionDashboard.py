import pylatex
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from .linear_text import *
import re
from .DashExceptions import ModelChoiceException
from dash.dependencies import Input, Output, State
from .Dashboard import Dashboard
from ..models.LinearRegressionModel import *


class PredictionDashboard(Dashboard):

    def _generate_layout(self):
        if self.settings['model'] == 'linreg':
            #          self.metrics_list = self.metric_to_method
            return LinearRegressionDashboard(self).get_layout()
        else:
            raise ModelChoiceException


class LinearRegressionDashboard(Dashboard):
    def __init__(self, predict: PredictionDashboard):
        Dashboard.__init__(self)
        self.predict = predict
        self.coord_list = []

    def get_layout(self):
        return self._generate_layout()

    def _generate_layout(self):
        metrics_list = []
        metrics_method = {
            'model_quality': self._generate_quality(),
            'signif': self._generate_signif(),
            'resid': self._generate_resid(),
            'equation': self._generate_equation(),
            'distrib_resid': self._generate_distrib()
        }
        for metric in metrics_method:
            if metric in self.predict.settings['metrics']:
                metrics_list.append(metrics_method[metric])

        # for metrics in self.predict.settings['metrics']:
        #    metrics_list.append(metrics_method[metrics])

        return html.Div(metrics_list)

    # графики
    def _generate_distrib(self):
        df_Y = self.predict.df_Y_test
        predict_Y = LinearRegressionModel.predict(self.predict.model, self.predict.df_X_test)

        # График распределения остатков
        fig_rasp_2 = go.Figure()
        df_ost_2 = pd.DataFrame({'Изначальный Y': df_Y, 'Предсказанный Y': predict_Y})
        fig_rasp_2 = px.scatter(df_ost_2, x="Изначальный Y", y="Предсказанный Y",
                                trendline="ols", trendline_color_override='red')
        fig_rasp_2.update_traces(marker_size=20)

        fig = go.Figure(data=go.Histogram(
            x=df_Y - predict_Y))
        fig.update_xaxes(title='Остатки')
        fig.update_layout(bargap=0.1)

        return html.Div([html.Div(html.H2(children='Графики остатков'), style={'text-align': 'center'}),
                         html.Div(html.H4(children='Гистограмма распределения остатков'), style={'text-align': 'center'}),
                         html.Div(dcc.Graph(id='Graph_ost_1', figure=fig)),
                         html.Div(html.H4(children='График соответствия предсказанных значений зависимой переменной '
                                                   'и исходных значений'), style={'text-align': 'center'}),
                         html.Div(dcc.Graph(id='Graph_ost_2', figure=fig_rasp_2)),
                         ], style={'margin': '50px'})

    # уравнение
    def _generate_equation(self):
        names = self.predict.settings['x']
        name_Y = self.predict.settings['y']
        b = self.predict.model.get_all_coef()
        uravnenie = LinearRegressionModel.uravnenie(self.predict.model, b, names, name_Y)
        df_X = self.predict.df_X_test
        b = self.predict.model.get_all_coef()

        def update_output(n_clicks, input1):
            number = len(self.coord_list)
            if n_clicks == 0 or input1 == 'Да':
                self.coord_list = []
                number = len(self.coord_list)
                return u'''Введите значение параметра "{}"'''.format(df_X.columns[0])
            if re.fullmatch(r'^([-+])?\d+([,.]\d+)?$', input1):
                number += 1
                if input1.find(',') > 0:
                    input1 = float(input1[0:input1.find(',')] + '.' + input1[input1.find(',') + 1:len(input1)])
                self.coord_list.append(float(input1))
                if len(self.coord_list) < len(df_X.columns):
                    return u'''Введите значение параметра  "{}".'''.format(df_X.columns[number])
                    # максимальное значение - len(df_X.columns)-1
                if len(self.coord_list) == len(df_X.columns):
                    number = -1
                    yzn = b[0]
                    for i in range(len(self.coord_list)):
                        yzn += self.coord_list[i] * b[i + 1]
                    return u'''Предсказанное значение равно {} \n Если желаете посчитать ещё для одного набор признаков
                    , напишите "Да".'''.format(round(yzn, 3))
            elif n_clicks > 0:
                return u'''Введено не число, введите значение параметра "{}" повторно.'''.format(df_X.columns[number])
            if number == -1 and input1 != 0 and input1 != 'Да' and input1 != '0':
                return u'''Если желаете посчитать ещё для {} набор признаков, напишите "Да".'''.format('одного')

        self.predict.app.callback(dash.dependencies.Output('output-state', 'children'),
                                  [dash.dependencies.Input('submit-button-state', 'n_clicks')],
                                  [dash.dependencies.State('input-1-state', 'value')])(update_output)
        return html.Div([html.Div(html.H2(children='Уравнение множественной регрессии'),
                                  style={'text-align': 'center'}),
                         html.Div([html.Div(dcc.Markdown(id='Markdown', children=uravnenie)),
                         html.Div(html.H4(children='Предсказание новых значений'),style={'text-align': 'center'}),
                                   dcc.Markdown(children='Чтобы получить значение зависимой переменной, '
                                                         'введите значение независимых признаков ниже:'),
                         dcc.Input(id='input-1-state', type='text', value=''),
                                   html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
                                   html.Div(id='output-state', children='')],
                                  style={'width': '78%', 'display': 'inline-block',
                                         'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                         'padding': '5px'})
                         ], style={'margin': '50px'})

    # качество модели
    def _generate_quality(self):
        df_result_1 = pd.DataFrame(columns=['Параметр', 'R', 'R2', 'R2adj', 'df', 'Fst', 'St.Error'])
        df_Y = self.predict.df_Y_test
        df_X = self.predict.df_X_test
        predict_Y = LinearRegressionModel.predict(self.predict.model, self.predict.df_X_test)
        mean_Y = LinearRegressionModel.get_mean(self.predict.model, df_Y)
        RSS = LinearRegressionModel.get_RSS(self.predict.model, predict_Y, mean_Y)
        de_fr = LinearRegressionModel.get_deg_fr(self.predict.model, self.predict.df_X_test)
        df_result_1.loc[1] = ['Значение', round(LinearRegressionModel.get_R(self.predict.model, df_Y, predict_Y), 3),
                              round(LinearRegressionModel.score(self.predict.model), 3),
                              round(LinearRegressionModel.get_R2_adj(self.predict.model, df_X, df_Y, predict_Y), 3),
                              str(str(LinearRegressionModel.get_deg_fr(self.predict.model, df_X)[0]) + '; ' +
                                  str(LinearRegressionModel.get_deg_fr(self.predict.model, df_X)[1])),
                              round(LinearRegressionModel.get_Fst(self.predict.model, df_X, df_Y, predict_Y), 3),
                              round(LinearRegressionModel.get_st_err(self.predict.model, RSS, de_fr), 3)
                              ]

        return html.Div([html.Div(html.H2(children='Критерии качества модели'), style={'text-align': 'center'}),
                         html.Div([html.Div(dash_table.DataTable(
                             id='table1',
                             columns=[{"name": i, "id": i} for i in df_result_1.columns],
                             data=df_result_1.to_dict('records')
                         ), style={'width': str(len(df_result_1.columns) * 10 - 10) + '%', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(markdown_linear_table1))],
                             style={'width': '78%', 'display': 'inline-block',
                                    'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'padding': '5px'})],
                         style={'margin': '50px'})

    # таблица остатков
    def _generate_resid(self):
        df_Y = self.predict.df_Y_test
        predict_Y = LinearRegressionModel.predict(self.predict.model, self.predict.df_X_test)
        mean_Y = sum(df_Y) / len(df_Y)
        TSS = LinearRegressionModel.get_TSS(self.predict.model, df_Y.tolist(), mean_Y)
        ESS = LinearRegressionModel.get_ESS(self.predict.model, df_Y.tolist(), predict_Y)
        RSS = LinearRegressionModel.get_RSS(self.predict.model, predict_Y, mean_Y)
        de_fr = LinearRegressionModel.get_deg_fr(self.predict.model, self.predict.df_X_test)
        d_1 = df_Y  # зависимый признак
        d_2 = predict_Y  # предсказанное значение
        d_3 = df_Y - predict_Y  # остатки
        d_4 = (predict_Y - mean_Y) / ((TSS / len(predict_Y)) ** 0.5)  # стандартизированные предсказанные значения
        d_5 = (df_Y - predict_Y) / ((ESS / len(df_Y)) ** 0.5)
        d_6 = df_Y * 0 + ((LinearRegressionModel.get_st_err(self.predict.model, RSS, de_fr) / len(df_Y)) ** 0.5)

        mean_list = []  # средние значения для каждого признака
        for i in range(self.predict.df_X_test.shape[1]):
            a = self.predict.df_X_test.iloc[:, i]
            mean_list.append(LinearRegressionModel.get_mean(self.predict.model, a))
        mah_df = []  # тут будут расстояния Махалонобиса для всех наблюдений
        cov_mat_2 = LinearRegressionModel.get_cov_matrix_2(self.predict.model,
                                                           self.predict.df_X_test)  # ков. матрица без единичного столбца
        for j in range(self.predict.df_X_test.shape[0]):
            aa = self.predict.df_X_test.iloc[j, :]  # строка с признаками
            meann = []  # список отличий от среднего
            for i in range(self.predict.df_X_test.shape[1]):
                meann.append(mean_list[i] - aa[i])
            mah_df.append(np.dot(np.dot(np.transpose(meann), cov_mat_2), meann))  # расстояние для наблюдения

        df_result_3 = pd.DataFrame({'Номер наблюдения': 0, 'Исходное значение признака': np.round(d_1, 3),
                                    'Рассчитанное значение признака': np.round(d_2, 3), 'Остатки': np.round(d_3, 3),
                                    'Стандартные предсказанные значения': np.round(d_4, 3),
                                    'Стандартизированные остатки': np.round(d_5, 3),
                                    'Стандартная ошибка предсказанного значения': np.round(d_6, 3),
                                    'Расстояние Махаланобиса': np.round(mah_df, 3)})
        df_result_3.iloc[:, 0] = [i + 1 for i in range(df_result_3.shape[0])]
        return html.Div([html.Div(html.H2(children='Таблица остатков'), style={'text-align': 'center'}),
                         html.Div([html.Div(dash_table.DataTable(
                             id='table3',
                             data=df_result_3.to_dict('records'),
                             columns=[{"name": i, "id": i} for i in df_result_3.columns],
                             # tooltip_header={i: i for i in df.columns}, # либо этот, либо тот что ниже
                             tooltip={i: {
                                 'value': i,
                                 'use_with': 'both'
                             } for i in df_result_3.columns},
                             style_header={
                                 'textDecoration': 'underline',
                                 'textDecorationStyle': 'dotted',
                             },
                             style_cell={
                                 'overflow': 'hidden',
                                 'textOverflow': 'ellipsis',
                                 'maxWidth': 0,  # len(df_result_3.columns)*5,
                             },
                             page_size=20,
                             fixed_rows={'headers': True},
                             style_table={'height': '330px', 'overflowY': 'auto'},
                             tooltip_delay=0,
                             tooltip_duration=None
                         ), style={'width': str(len(df_result_3.columns) * 10 - 10) + '%', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(markdown_linear_table3))],
                             style={'width': '78%', 'display': 'inline-block',
                                    'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'padding': '5px'})
                         ],style={'margin': '50px'})

    # таблица критериев значимости переменных
    def _generate_signif(self):
        df_Y = self.predict.df_Y_test
        predict_Y = LinearRegressionModel.predict(self.predict.model, self.predict.df_X_test)
        mean_Y = LinearRegressionModel.get_mean(self.predict.model, df_Y.tolist())
        TSS = LinearRegressionModel.get_TSS(self.predict.model, df_Y.tolist(), mean_Y)
        de_fr = LinearRegressionModel.get_deg_fr(self.predict.model, self.predict.df_X_test)
        b = self.predict.model.get_all_coef()
        df_column = list(self.predict.df_X_test.columns)
        df_column.insert(0, 'Параметр')
        df_result_2 = pd.DataFrame(columns=df_column)
        t_st = LinearRegressionModel.t_stat(self.predict.model, self.predict.df_X_test, self.predict.df_Y_test, predict_Y, de_fr,
                                            b)
        cov_mat = LinearRegressionModel.get_cov_matrix(self.predict.model, self.predict.df_X_test)
        st_er_coef = LinearRegressionModel.st_er_coef(self.predict.model, self.predict.df_Y_test, predict_Y, cov_mat)
        p_values = LinearRegressionModel.p_values(self.predict.model, self.predict.df_X_test, t_st)
        b_st = LinearRegressionModel.st_coef(self.predict.model, self.predict.df_X_test, TSS, b)

        res_b = ['b']
        list_b = list(b)
        for j in range(1, len(list_b)):
            res_b.append(round(list_b[j], 3))
        df_result_2.loc[1] = res_b

        res_bst = ['b_st']
        list_bst = list(b_st)
        for j in range(len(list_bst)):
            res_bst.append(round(list_bst[j], 3))
        df_result_2.loc[2] = res_bst

        res_errb = ['St.Error b']
        st_er_b = list(st_er_coef)
        for j in range(1, len(st_er_b)):
            res_errb.append(round(st_er_b[j], 3))
        df_result_2.loc[3] = res_errb

        res_tst = ['t-критерий']
        for j in range(1, len(t_st)):
            res_tst.append(round(t_st[j], 3))
        df_result_2.loc[4] = res_tst

        res_pval = ['p-value']
        for j in range(1, len(t_st)):
            res_pval.append(round(p_values[j], 3))
        df_result_2.loc[5] = res_pval

        return html.Div([html.Div(html.H2(children='Критерии значимости переменных'), style={'text-align': 'center'}),
                         html.Div([html.Div(dash_table.DataTable(
                             id='table2',
                             columns=[{"name": i, "id": i} for i in df_result_2.columns],
                             data=df_result_2.to_dict('records'),
                             style_cell={
                                 'overflow': 'hidden',
                                 'textOverflow': 'ellipsis',
                                 'maxWidth': 0,  # len(df_result_3.columns)*5,
                             },
                         ), style={'width': str(len(df_result_2.columns) * 10) + '%', 'display': 'inline-block'}),
                                 html.Div(dcc.Markdown(markdown_linear_table2))], #style={'margin': '50px'},
                                 style={'width': '78%', 'display': 'inline-block',
                                        'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'padding': '5px'})
                         ], style={'margin': '50px'})

