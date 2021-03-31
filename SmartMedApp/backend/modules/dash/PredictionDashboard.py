import re

import pylatex

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import sklearn.metrics as sm
import pandas as pd
import scipy.stats as sps
from scipy.sparse import issparse
from sklearn.feature_selection import chi2

import plotly.graph_objects as go
import plotly.express as px

from .text.linear_text import *
from .text.roc_text import *

from .DashExceptions import ModelChoiceException
from .Dashboard import Dashboard

from ..models.LinearRegressionModel import *
from ..models.LogisticRegressionModel import *


class PredictionDashboard(Dashboard):

    def _generate_layout(self):
        if self.settings['model'] == 'linreg':
            return LinearRegressionDashboard(self).get_layout()
        elif self.settings['model'] == 'logreg':
            return LogisticRegressionDashboard(self).get_layout()
        elif self.settings['model'] == 'roc':
            return ROC(self).get_layout()
        elif self.settings['model'] == 'polynomreg':
            return PolynomRegressionDashboard(self).get_layout()
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

        return html.Div([
            html.Div(html.H1(children='Множественная регрессия'), style={'text-align': 'center'}),
            html.Div(metrics_list)])

    # графики
    def _generate_distrib(self):
        df_Y = self.predict.df_Y_test
        predict_Y = LinearRegressionModel.predict(
            self.predict.model, self.predict.df_X_test)

        # График распределения остатков
        fig_rasp_2 = go.Figure()
        df_ost_2 = pd.DataFrame(
            {'Изначальный Y': df_Y, 'Предсказанный Y': predict_Y})
        fig_rasp_2 = px.scatter(df_ost_2, x="Изначальный Y", y="Предсказанный Y",
                                trendline="ols", trendline_color_override='red', labels='Данные')
        fig_rasp_2.update_traces(marker_size=20)

        fig = go.Figure(
            data=go.Histogram(
                x=df_Y - predict_Y,
                name='Остатки'
            )
        )

        fig.add_trace(
            go.Histogram(
                x=np.random.normal(0, 1, len(df_Y)),
                name='Нормальное распределение'
            )
        )
        fig.update_xaxes(title='Остатки')
        fig.update_layout(bargap=0.1)

        # специфичность

        residuals = df_Y - predict_Y
        num_divisions = residuals.shape[0] + 1
        quantiles = np.arange(1, residuals.shape[0]) / num_divisions

        qq_x_data = sps.norm.ppf(quantiles)
        qq_y_data = np.sort(residuals)

        line_x0 = sps.norm.ppf(0.25)
        line_x1 = sps.norm.ppf(0.75)
        line_y0 = np.quantile(residuals, 0.25)
        line_y1 = np.quantile(residuals, 0.75)
        slope = (line_y1 - line_y0) / (line_x1 - line_x0)
        line_intercept = line_y1 - (slope * line_x1)
        x_range_line = np.arange(-3, 3, 0.001)
        y_values_line = (slope * x_range_line) + line_intercept
        fig_qqplot = go.Figure()
        fig_qqplot.add_trace(
            go.Scatter(
                x=qq_x_data,
                y=qq_y_data,
                mode='markers',
                marker={'color': 'blue'},
                name='Остатки')
        )
        fig_qqplot.add_trace(
            go.Scatter(
                x=x_range_line,
                y=y_values_line,
                mode='lines',
                marker={'color': 'red'},
                name='Нормальное распределение'))
        fig_qqplot['layout'].update(
            xaxis={
                'title': 'Теоретические квантили',
                'zeroline': True},
            yaxis={
                'title': 'Экспериментальные квантили'},
            showlegend=True,
        )

        return html.Div([html.Div(html.H2(children='Графики остатков'), style={'text-align': 'center'}),
                         html.Div([
                             html.Div(
                                 html.H4(children='Гистограмма распределения остатков'), style={'text-align': 'center'}),
                             html.Div(dcc.Graph(id='Graph_ost_1', figure=fig),
                                      style={'text-align': 'center', 'width': '78%', 'display': 'inline-block',
                                             'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid'}),
                         ], style={'margin': '50px'}),

                         html.Div([
                             html.Div(
                                 html.H4(children='График соответствия предсказанных значений зависимой переменной '
                                                   'и исходных значений'), style={'text-align': 'center'}),
                             html.Div(dcc.Graph(id='Graph_ost_2', figure=fig_rasp_2),
                                      style={'text-align': 'center', 'width': '78%', 'display': 'inline-block',
                                             'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid'}),
                             html.Div(dcc.Markdown(markdown_graph))
                         ], style={'margin': '50px'}),

                         html.Div([
                             html.Div(
                                 html.H4(children='График квантиль-квантиль'), style={'text-align': 'center'}),
                             html.Div(dcc.Graph(id='graph_qqplot', figure=fig_qqplot),
                                      style={'text-align': 'center', 'width': '78%', 'display': 'inline-block',
                                             'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid'}),
                         ], style={'margin': '50px'}),

                         ], style={'margin': '50px'})

    # уравнение
    def _generate_equation(self):
        names = self.predict.settings['x']
        name_Y = self.predict.settings['y']
        b = self.predict.model.get_all_coef()
        uravnenie = LinearRegressionModel.uravnenie(
            self.predict.model, b, names, name_Y)
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
                    input1 = float(input1[0:input1.find(
                        ',')] + '.' + input1[input1.find(',') + 1:len(input1)])
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
                                  [dash.dependencies.Input(
                                      'submit-button-state', 'n_clicks')],
                                  [dash.dependencies.State('input-1-state', 'value')])(update_output)
        return html.Div([html.Div(html.H2(children='Уравнение множественной регрессии'),
                                  style={'text-align': 'center'}),
                         html.Div([html.Div(dcc.Markdown(id='Markdown', children=uravnenie)),
                                   html.Div(html.H4(children='Предсказание новых значений'),
                                            style={'text-align': 'center'}),
                                   dcc.Markdown(children='Чтобы получить значение зависимой переменной, '
                                                         'введите значение независимых признаков ниже:'),
                                   dcc.Input(id='input-1-state',
                                             type='text', value=''),
                                   html.Button(id='submit-button-state',
                                               n_clicks=0, children='Submit'),
                                   html.Div(id='output-state', children='')],
                                  style={'width': '78%', 'display': 'inline-block',
                                         'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                         'padding': '5px'})
                         ], style={'margin': '50px'})

    # качество модели
    def _generate_quality(self):
        df_result_1 = pd.DataFrame(
            columns=['Параметр', 'R', 'R2', 'R2adj', 'df', 'Fst', 'St.Error'])
        df_Y = self.predict.df_Y_test
        df_X = self.predict.df_X_test
        predict_Y = LinearRegressionModel.predict(
            self.predict.model, self.predict.df_X_test)
        mean_Y = LinearRegressionModel.get_mean(self.predict.model, df_Y)
        RSS = LinearRegressionModel.get_RSS(
            self.predict.model, predict_Y, mean_Y)
        de_fr = LinearRegressionModel.get_deg_fr(
            self.predict.model, self.predict.df_X_test)
        df_result_1.loc[1] = ['Значение', round(LinearRegressionModel.get_R(self.predict.model, df_Y, predict_Y), 3),
                              round(LinearRegressionModel.score(
                                  self.predict.model), 3),
                              round(LinearRegressionModel.get_R2_adj(
                                  self.predict.model, df_X, df_Y, predict_Y), 3),
                              str(str(LinearRegressionModel.get_deg_fr(self.predict.model, df_X)[0]) + '; ' +
                                  str(LinearRegressionModel.get_deg_fr(self.predict.model, df_X)[1])),
                              round(LinearRegressionModel.get_Fst(
                                  self.predict.model, df_X, df_Y, predict_Y), 3),
                              round(LinearRegressionModel.get_st_err(
                                  self.predict.model, RSS, de_fr), 3)
                              ]

        return html.Div([html.Div(html.H2(children='Критерии качества модели'), style={'text-align': 'center'}),
                         html.Div([html.Div(dash_table.DataTable(
                             id='table1',
                             columns=[{"name": i, "id": i}
                                      for i in df_result_1.columns],
                             data=df_result_1.to_dict('records'),
                             export_format='xlsx'
                         ), style={'width': str(len(df_result_1.columns) * 8 - 10) + '%', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(markdown_linear_table1))],
                             style={'width': '78%', 'display': 'inline-block',
                                    'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'padding': '5px'})],
                        style={'margin': '50px'})

    # таблица остатков
    def _generate_resid(self):
        df_Y = self.predict.df_Y_test
        predict_Y = LinearRegressionModel.predict(
            self.predict.model, self.predict.df_X_test)
        mean_Y = sum(df_Y) / len(df_Y)
        TSS = LinearRegressionModel.get_TSS(
            self.predict.model, df_Y.tolist(), mean_Y)
        ESS = LinearRegressionModel.get_ESS(
            self.predict.model, df_Y.tolist(), predict_Y)
        RSS = LinearRegressionModel.get_RSS(
            self.predict.model, predict_Y, mean_Y)
        de_fr = LinearRegressionModel.get_deg_fr(
            self.predict.model, self.predict.df_X_test)
        d_1 = df_Y  # зависимый признак
        d_2 = predict_Y  # предсказанное значение
        d_3 = df_Y - predict_Y  # остатки
        # стандартизированные предсказанные значения
        d_4 = (predict_Y - mean_Y) / ((TSS / len(predict_Y)) ** 0.5)
        d_5 = (df_Y - predict_Y) / ((ESS / len(df_Y)) ** 0.5)
        d_6 = df_Y * 0 + \
            ((LinearRegressionModel.get_st_err(
                self.predict.model, RSS, de_fr) / len(df_Y)) ** 0.5)

        mean_list = []  # средние значения для каждого признака
        for i in range(self.predict.df_X_test.shape[1]):
            a = self.predict.df_X_test.iloc[:, i]
            mean_list.append(
                LinearRegressionModel.get_mean(self.predict.model, a))
        mah_df = []  # тут будут расстояния Махалонобиса для всех наблюдений
        cov_mat_2 = LinearRegressionModel.get_cov_matrix_2(self.predict.model,
                                                           self.predict.df_X_test)  # ков. матрица без единичного столбца
        for j in range(self.predict.df_X_test.shape[0]):
            aa = self.predict.df_X_test.iloc[j, :]  # строка с признаками
            meann = []  # список отличий от среднего
            for i in range(self.predict.df_X_test.shape[1]):
                meann.append(mean_list[i] - aa[i])
            # расстояние для наблюдения
            mah_df.append(
                np.dot(np.dot(np.transpose(meann), cov_mat_2), meann))

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
                             columns=[{"name": i, "id": i}
                                      for i in df_result_3.columns],
                             # tooltip_header={i: i for i in df.columns}, #
                             # либо этот, либо тот что ниже
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

                             # asdf
                             page_size=20,
                             fixed_rows={'headers': True},
                             style_table={'height': '330px',
                                          'overflowY': 'auto'},
                             tooltip_delay=0,
                             tooltip_duration=None,
                             export_format='xlsx'
                         ), style={'width': str(len(df_result_3.columns) * 8 - 10) + '%', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(markdown_linear_table3))],
                             style={'width': '78%', 'display': 'inline-block',
                                    'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'padding': '5px'})
                         ], style={'margin': '50px'})

    # таблица критериев значимости переменных
    def _generate_signif(self):
        df_Y = self.predict.df_Y_test
        predict_Y = LinearRegressionModel.predict(
            self.predict.model, self.predict.df_X_test)
        mean_Y = LinearRegressionModel.get_mean(
            self.predict.model, df_Y.tolist())
        TSS = LinearRegressionModel.get_TSS(
            self.predict.model, df_Y.tolist(), mean_Y)
        de_fr = LinearRegressionModel.get_deg_fr(
            self.predict.model, self.predict.df_X_test)
        b = self.predict.model.get_all_coef()
        df_column = list(self.predict.df_X_test.columns)
        df_column.insert(0, 'Параметр')
        df_result_2 = pd.DataFrame(columns=df_column)
        t_st = LinearRegressionModel.t_stat(self.predict.model, self.predict.df_X_test, self.predict.df_Y_test,
                                            predict_Y, de_fr,
                                            b)
        cov_mat = LinearRegressionModel.get_cov_matrix(
            self.predict.model, self.predict.df_X_test)
        st_er_coef = LinearRegressionModel.st_er_coef(
            self.predict.model, self.predict.df_Y_test, predict_Y, cov_mat)
        p_values = LinearRegressionModel.p_values(
            self.predict.model, self.predict.df_X_test, t_st)
        b_st = LinearRegressionModel.st_coef(
            self.predict.model, self.predict.df_X_test, TSS, b)

        res_b = []
        list_b = list(b)
        for j in range(1, len(list_b)):
            res_b.append(round(list_b[j], 3))

        res_bst = []
        list_bst = list(b_st)
        for j in range(len(list_bst)):
            res_bst.append(round(list_bst[j], 3))

        res_errb = []
        st_er_b = list(st_er_coef)
        for j in range(1, len(st_er_b)):
            res_errb.append(round(st_er_b[j], 3))

        res_tst = []
        for j in range(1, len(t_st)):
            res_tst.append(round(t_st[j], 3))

        res_pval = []
        for j in range(1, len(t_st)):
            res_pval.append(round(p_values[j], 3))

        df_result_2 = pd.DataFrame({'Название переменной': self.predict.df_X.columns.tolist(),
                                    'b': res_b,
                                    'b_st': res_bst,
                                    'St.Error b': res_errb,
                                    't-критерий': res_tst,
                                    'p-value': res_pval})

        return html.Div([html.Div(html.H2(children='Критерии значимости переменных'), style={'text-align': 'center'}),
                         html.Div([html.Div(dash_table.DataTable(
                             id='table2',
                             columns=[{"name": i, "id": i}
                                      for i in df_result_2.columns],
                             data=df_result_2.to_dict('records'),
                             style_table={'textOverflowX': 'ellipsis', },
                             tooltip={i: {
                                 'value': i,
                                 'use_with': 'both'
                             } for i in df_result_2.columns},
                             tooltip_data=[
                                 {
                                     column: {'value': str(value), 'type': 'markdown'}
                                     for column, value in row.items()
                                 } for row in df_result_2.to_dict('records')
                             ],
                             style_header={
                                 'textDecoration': 'underline',
                                 'textDecorationStyle': 'dotted',
                             },
                             style_cell={
                                 'overflow': 'hidden',
                                 'textOverflow': 'ellipsis',
                                 'maxWidth': 0,  # len(df_result_3.columns)*5,
                             },
                             export_format='xlsx'

                         ), style={'width': str(len(df_result_2.columns) * 6) + '%', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(markdown_linear_table2))],  # style={'margin': '50px'},
                             style={'width': '78%', 'display': 'inline-block',
                                    'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'padding': '5px'})
                         ], style={'margin': '50px'})


class LogisticRegressionDashboard(Dashboard):
    def __init__(self, predict: PredictionDashboard):
        Dashboard.__init__(self)
        self.predict = predict
        self.coord_list = []

    def get_layout(self):
        return self._generate_layout()

    def _generate_layout(self):
        metrics_list = [self._generate_matrix()]
        metrics_method = {
            'model_quality': self._generate_quality(),
            'signif': self._generate_signif(),
            'resid': self._generate_resid(),
        }
        for metric in metrics_method:
            if metric in self.predict.settings['metrics']:
                metrics_list.append(metrics_method[metric])
        # for metrics in self.predict.settings['metrics']:
        #    metrics_list.append(metrics_method[metrics])
        df_X = self.predict.df_X_test
        if np.any((df_X.data if issparse(df_X) else df_X) < 0):
            return html.Div([html.Div(html.H1(children='Логистическая регрессия'), style={'text-align': 'center'}),
                             html.Div(dcc.Markdown(markdown_error),
                                      style={'width': '78%', 'display': 'inline-block',
                                             'border-color': 'rgb(220, 220, 220)',
                                             'border-style': 'solid', 'padding': '5px'})],
                            style={'margin': '50px'})
        else:
            return html.Div([
                html.Div(html.H1(children='Логистическая регрессия'), style={'text-align': 'center'}),
                html.Div(metrics_list)])

    def _generate_matrix(self):
        df_X = self.predict.df_X_test
        y_true = self.predict.df_Y_test
        y_pred = LogisticRegressionModel.predict(self.predict.model, df_X)
        TN, FP, FN, TP = sm.confusion_matrix(y_true, y_pred).ravel()
        df_matrix = pd.DataFrame(columns=['y_pred\y_true', 'True', 'False'])
        df_matrix.loc[1] = ['True', TP, FP]
        df_matrix.loc[2] = ['False', FN, TN]
        return html.Div([html.Div(html.H2(children='Матрица классификации'), style={'text-align': 'center'}),
                         html.Div([html.Div(dash_table.DataTable(
                             id='table_matrix',
                             columns=[{"name": i, "id": i} for i in df_matrix.columns],
                             data=df_matrix.to_dict('records'),
                             tooltip={i: {
                                 'value': i,
                                 'use_with': 'both'
                             } for i in df_matrix.columns},
                             export_format='csv',
                             style_header={
                                 'textDecoration': 'underline',
                                 'textDecorationStyle': 'dotted',
                             },
                         ), style={'width': str(len(df_matrix.columns) * 8) + '%', 'display': 'inline-block'}),
                             #html.Div(dcc.Markdown(markdown_linear_table1))
                         ],
                             style={'width': '78%', 'display': 'inline-block',
                                    'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'padding': '5px'})],
                        style={'margin': '50px'})

    # качество модели
    def _generate_quality(self):
        df_result_1 = pd.DataFrame(columns=['Критерий', 'Хи-квадрат', 'Степень свободы', 'p-value'])
        df_Y = self.predict.df_Y_test
        df_X = self.predict.df_X_test
        chi_table = chi2(df_X, df_Y)
        for i in range(len(chi_table[0])):
                df_result_1.loc[i+1] = [df_X.columns.tolist()[i],
                                    round(chi_table[0][i], 3),
                                    len(df_Y),
                                    round(chi_table[1][i], 3)
                                    ]

        return html.Div([html.Div(html.H2(children='Критерии качества модели'), style={'text-align': 'center'}),
                            html.Div([html.Div(dash_table.DataTable(
                                id='table1',
                                columns=[{"name": i, "id": i} for i in df_result_1.columns],
                                data=df_result_1.to_dict('records'),
                                tooltip={i: {
                                    'value': i,
                                    'use_with': 'both'
                                } for i in df_result_1.columns},
                                style_header={
                                    'textDecoration': 'underline',
                                    'textDecorationStyle': 'dotted',
                                },
                                export_format='csv'
                            ), style={'width': str(len(df_result_1.columns) * 8) + '%', 'display': 'inline-block'}),
                                html.Div(dcc.Markdown(markdown_linear_table1))],
                                style={'width': '78%', 'display': 'inline-block', 'border-color': 'rgb(220, 220, 220)',
                                        'border-style': 'solid', 'padding': '5px'})],
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
                         ), style={'width': str(len(df_result_3.columns) * 8) + '%', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(markdown_linear_table3))],
                             style={'width': '78%', 'display': 'inline-block',
                                    'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'padding': '5px'})
                         ], style={'margin': '50px'})

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
        t_st = LinearRegressionModel.t_stat(self.predict.model, self.predict.df_X_test, self.predict.df_Y_test,
                                            predict_Y, de_fr,
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
                             tooltip={i: {
                                 'value': i,
                                 'use_with': 'both'
                             } for i in df_result_2.columns},
                             tooltip_data=[
                                 {
                                     column: {'value': str(value), 'type': 'markdown'}
                                     for column, value in row.items()
                                 } for row in df_result_2.to_dict('records')
                             ],
                         ), style={'width': str(len(df_result_2.columns) * 10) + '%', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(markdown_linear_table2))],  # style={'margin': '50px'},
                             style={'width': '78%', 'display': 'inline-block',
                                    'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'padding': '5px'})
                         ], style={'margin': '50px'})


class PolynomRegressionDashboard(Dashboard):
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

        return html.Div([
            html.Div(html.H1(children='Полиномиальная регрессия'), style={'text-align': 'center'}),
            html.Div(metrics_list)])

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

        fig = go.Figure(
            data=go.Histogram(
                    x=df_Y - predict_Y,
                    name='Остатки'
            )
        )

        fig.add_trace(
            go.Histogram(
                x=np.random.normal(0, 1, len(df_Y)),
                name='Нормальное распределение'
            )
        )
        fig.update_xaxes(title='Остатки')
        fig.update_layout(bargap=0.1)

        # специфичность

        residuals = df_Y - predict_Y
        num_divisions = residuals.shape[0] + 1
        quantiles = np.arange(1, residuals.shape[0]) / num_divisions

        qq_x_data = sps.norm.ppf(quantiles)
        qq_y_data = np.sort(residuals)

        line_x0 = sps.norm.ppf(0.25)
        line_x1 = sps.norm.ppf(0.75)
        line_y0 = np.quantile(residuals, 0.25)
        line_y1 = np.quantile(residuals, 0.75)
        slope = (line_y1 - line_y0) / (line_x1 - line_x0)
        line_intercept = line_y1 - (slope * line_x1)
        x_range_line = np.arange(-3, 3, 0.001)
        y_values_line = (slope * x_range_line) + line_intercept
        fig_qqplot = go.Figure()
        fig_qqplot.add_trace(
            go.Scatter(
                x=qq_x_data,
                y=qq_y_data,
                mode='markers',
                marker={'color': 'blue'},
                name='Остатки')
        )
        fig_qqplot.add_trace(
            go.Scatter(
                x=x_range_line,
                y=y_values_line,
                mode='lines',
                marker={'color': 'red'},
                name='Нормальное распределение'))
        fig_qqplot['layout'].update(
            xaxis={
                'title': 'Теоретические квантили',
                'zeroline': True},
            yaxis={
                'title': 'Экспериментальные квантили'},
            showlegend=True,

        )

        return html.Div([html.Div(html.H2(children='Графики остатков'), style={'text-align': 'center'}),
                         html.Div([
                             html.Div(
                                 html.H4(children='Гистограмма распределения остатков'),
                                 style={'text-align': 'center'}),
                             html.Div(dcc.Graph(id='Graph_ost_1', figure=fig),
                                      style={'text-align': 'center', 'width': '78%', 'display': 'inline-block',
                                             'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid'}),
                         ], style={'margin': '50px'}),

                         html.Div([
                             html.Div(
                                 html.H4(children='График соответствия предсказанных значений зависимой переменной '
                                                  'и исходных значений'), style={'text-align': 'center'}),
                             html.Div([dcc.Graph(id='Graph_ost_2', figure=fig_rasp_2),
                                       html.Div(dcc.Markdown(markdown_graph))],
                                      style={'width': '78%', 'display': 'inline-block',
                                             'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid',
                                             'padding': '5px'}),
                         ], style={'margin': '50px'}),


                         html.Div([
                             html.Div(html.H4(children='График квантиль-квантиль'), style={'text-align': 'center'}),
                             html.Div(dcc.Graph(id='graph_qqplot', figure=fig_qqplot),
                                      style={'text-align': 'center', 'width': '78%', 'display': 'inline-block',
                                             'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid'}),
                         ], style={'margin': '50px'}),

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

        #def update_input(n_clicks):
        #    return ' '

        #self.predict.app.callback(dash.dependencies.Output('input-1-state', 'value'),
        #                            [dash.dependencies.Input('submit-button-state', 'n_clicks')])(update_input)

        return html.Div([html.Div(html.H2(children='Уравнение множественной регрессии'),
                                  style={'text-align': 'center'}),
                         html.Div([html.Div(dcc.Markdown(id='Markdown', children=uravnenie)),
                                   html.Div(html.H4(children='Предсказание новых значений'),
                                            style={'text-align': 'center'}),
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
                             data=df_result_1.to_dict('records'),
                             export_format='xlsx'
                         ), style={'width': str(len(df_result_1.columns) * 8 - 10) + '%', 'display': 'inline-block'}),
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
        #mah_df = []  # тут будут расстояния Махалонобиса для всех наблюдений
        #cov_mat_2 = LinearRegressionModel.get_cov_matrix_2(self.predict.model,
        #                                                   self.predict.df_X_test)  # ков. матрица без единичного столбца
        #for j in range(self.predict.df_X_test.shape[0]):
        #    aa = self.predict.df_X_test.iloc[j, :]  # строка с признаками
        #    meann = []  # список отличий от среднего
        #    for i in range(self.predict.df_X_test.shape[1]):
        #        meann.append(mean_list[i] - aa[i])
        #    mah_df.append(np.dot(np.dot(np.transpose(meann), cov_mat_2), meann))  # расстояние для наблюдения

        df_result_3 = pd.DataFrame({'Номер наблюдения': 0, 'Исходное значение признака': np.round(d_1, 3),
                                    'Рассчитанное значение признака': np.round(d_2, 3), 'Остатки': np.round(d_3, 3),
                                    'Стандартные предсказанные значения': np.round(d_4, 3),
                                    'Стандартизированные остатки': np.round(d_5, 3),
                                    'Стандартная ошибка предсказанного значения': np.round(d_6, 3)})
                                    #'Расстояние Махаланобиса': np.round(mah_df, 3)})
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

                             # asdf
                             page_size=20,
                             fixed_rows={'headers': True},
                             style_table={'height': '330px', 'overflowY': 'auto'},
                             tooltip_delay=0,
                             tooltip_duration=None,
                             export_format='xlsx'
                         ), style={'width': str(len(df_result_3.columns) * 8 - 10) + '%', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(markdown_linear_table3))],
                             style={'width': '78%', 'display': 'inline-block',
                                    'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'padding': '5px'})
                         ], style={'margin': '50px'})

    # таблица критериев значимости переменных
    def _generate_signif(self):
        df_Y = self.predict.df_Y_test
        predict_Y = LinearRegressionModel.predict(self.predict.model, self.predict.df_X_test)
        mean_Y = LinearRegressionModel.get_mean(self.predict.model, df_Y.tolist())
        TSS = LinearRegressionModel.get_TSS(self.predict.model, df_Y.tolist(), mean_Y)
        de_fr = LinearRegressionModel.get_deg_fr(self.predict.model, self.predict.df_X_test)
        b = self.predict.model.get_all_coef()

        t_st = LinearRegressionModel.t_stat(self.predict.model, self.predict.df_X_test, self.predict.df_Y_test,
                                            predict_Y, de_fr,
                                            b)
        cov_mat = LinearRegressionModel.get_cov_matrix(self.predict.model, self.predict.df_X_test)
        st_er_coef = LinearRegressionModel.st_er_coef(self.predict.model, self.predict.df_Y_test, predict_Y, cov_mat)
        p_values = LinearRegressionModel.p_values(self.predict.model, self.predict.df_X_test, t_st)
        b_st = LinearRegressionModel.st_coef(self.predict.model, self.predict.df_X_test, TSS, b)

        res_b = []
        list_b = list(b)
        for j in range(1, len(list_b)):
            res_b.append(round(list_b[j], 3))

        res_bst = []
        list_bst = list(b_st)
        for j in range(len(list_bst)):
            res_bst.append(round(list_bst[j], 3))

        res_errb = []
        st_er_b = list(st_er_coef)
        for j in range(1, len(st_er_b)):
            res_errb.append(round(st_er_b[j], 3))

        res_tst = []
        for j in range(1, len(t_st)):
            res_tst.append(round(t_st[j], 3))

        res_pval = []
        for j in range(1, len(t_st)):
            res_pval.append(round(p_values[j], 3))

        df_result_2 = pd.DataFrame({'Название переменной': self.predict.df_X.columns.tolist(),
                                    'b': res_b,
                                    'b_st': res_bst,
                                    'St.Error b': res_errb,
                                    't-критерий': res_tst,
                                    'p-value': res_pval})
        # 'Расстояние Махаланобиса': np.round(mah_df, 3)})

        return html.Div([html.Div(html.H2(children='Критерии значимости переменных'), style={'text-align': 'center'}),
                         html.Div([html.Div(dash_table.DataTable(
                             id='table2',
                             columns=[{"name": i, "id": i} for i in df_result_2.columns],
                             data=df_result_2.to_dict('records'),
                             style_table={'textOverflowX': 'ellipsis', },
                             tooltip={i: {
                                 'value': i,
                                 'use_with': 'both'
                             } for i in df_result_2.columns},

                             style_header={
                                 'textDecoration': 'underline',
                                 'textDecorationStyle': 'dotted',
                             },
                             style_cell={
                                 'overflow': 'hidden',
                                 'textOverflow': 'ellipsis',
                                 'maxWidth': 0,  # len(df_result_3.columns)*5,
                             },
                             export_format='xlsx',
                             tooltip_data=[
                                 {
                                     column: {'value': str(value), 'type': 'markdown'}
                                     for column, value in row.items()
                                 } for row in df_result_2.to_dict('records')
                             ],

                         ), style={'width': str(len(df_result_2.columns) * 6) + '%', 'display': 'inline-block'}),
                             html.Div(dcc.Markdown(markdown_linear_table2))],  # style={'margin': '50px'},
                             style={'width': '78%', 'display': 'inline-block',
                                    'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'padding': '5px'})
                         ], style={'margin': '50px'})


class ROC(Dashboard):

    def __init__(self, predict: PredictionDashboard):
        Dashboard.__init__(self)
        self.predict = predict

        # тут вложенными списками будут значения для каждой переменной
        self.dx_list = []

        self.tp_list = []
        self.tn_list = []
        self.fp_list = []
        self.fn_list = []

        self.se_list = []  # чувствительность
        self.sp_list = []  # специфичность
        self.inv_sp_list = []  # 1-специфичность

    def get_layout(self):
        return self._generate_layout()

    def _generate_layout(self):
        if 'classificators_comparison' in self.predict.settings['metrics']:
            metrics_list = [
                self._generate_dashboard(),
                self._generate_comparison()]
        else:
            metrics_list = [self._generate_dashboard()]

        return html.Div([
            html.Div(html.H1(children='ROC-анализ'), style={'text-align': 'center'}),
            html.Div(metrics_list)])

    def _generate_metrics(self, ind):
        threshold = 1
        t_ind = 0
        for i in range(len(self.se_list[ind])):
            if threshold > abs(self.se_list[ind][i] - self.sp_list[ind][i]):
                threshold = abs(self.se_list[ind][i] - self.sp_list[ind][i])
                t_ind = i
        threshold = round(threshold, 3)
        TPR = round(self.tp_list[ind][
                    t_ind] / (self.tp_list[ind][t_ind] + self.fn_list[ind][t_ind]), 3)
        PPV = round(self.tp_list[ind][
                    t_ind] / (self.tp_list[ind][t_ind] + self.fp_list[ind][t_ind]), 3)
        print(ind, TPR, PPV)
        accuracy = round((self.tp_list[ind][t_ind] + self.tn_list[ind][t_ind]) / (
            self.tp_list[ind][t_ind] + self.fn_list[ind][t_ind] + self.tn_list[ind][t_ind] + self.fp_list[ind][
                t_ind]), 3)
        f_measure = round(2 * self.tp_list[ind][t_ind] / (
            2 * self.tp_list[ind][t_ind] + self.fn_list[ind][t_ind] + self.fp_list[ind][t_ind]), 3)
        auc = 0
        for i in range(len(self.sp_list[ind]) - 1):
            auc += (self.se_list[ind][i] + self.se_list[ind][i + 1]) * (
                self.inv_sp_list[ind][i + 1] - self.inv_sp_list[ind][i]) / 2
        auc = round(abs(auc), 3)
        dov_int = (np.var(self.se_list[ind]) /
                   (len(self.se_list[ind]) * (len(self.se_list[ind]) - 1))) ** 0.5
        dov_int_1 = round((self.se_list[ind][t_ind] - 1.96 * dov_int), 3)
        dov_int_2 = round((self.se_list[ind][t_ind] + 1.96 * dov_int), 3)
        df_ost_2 = pd.DataFrame(
            columns=['Параметр', 'Threshold', 'Оптимальный порог', 'Полнота',
                     'Точность', 'Accuracy', 'F-мера', 'Доверительный интервал', 'AUC'])
        df_ost_2.loc[1] = ['Значение', threshold, round(self.dx_list[ind][t_ind], 3), TPR, PPV, accuracy,
                           f_measure, str(str(dov_int_1) + ';' + str(dov_int_2)), auc]

        return df_ost_2

    def _generate_graphs(self):
        # df_ost_2 = pd.DataFrame(
        #    {'dx': self.dx_list, 'tp': self.tp_list, 'fp': self.fp_list, 'fn': self.fn_list, 'tn': self.tn_list,
        #     'sp': self.sp_list, 'se': self.se_list})
        #       fig_rasp_2 = px.scatter(df_ost_2, x="sp", y="se")
        #       fig_rasp_2.update_traces(marker_size=20)
        fig_rasp_2 = go.Figure()
        # px.scatter(df_ost_2, x="dx", y="se")
        fig_rasp_2.add_trace(
            go.Scatter(
                x=self.sp_list,
                y=self.se_list,
                #               xaxis='se',
                #               yaxis='1-sp',
                mode="lines+markers",
                line=go.scatter.Line(color="red"),
                showlegend=True)
        )
        fig_rasp_2.update_traces(marker_size=10)
        return html.Div([html.Div(html.H4(children='ROC'), style={'text-align': 'center'}),
                         html.Div(dcc.Graph(id='graph_ROC', figure=fig_rasp_2)),
                         ], style={'margin': '50px'})

    def _generate_dots(self, ind):
        df_ost_2 = pd.DataFrame(
            {'dx': [round(self.dx_list[ind][i], 3) for i in range(len(self.dx_list[ind]))], 'tp': self.tp_list[ind],
             'fp': self.fp_list[ind], 'fn': self.fn_list[ind], 'tn': self.tn_list[ind], 'sp': self.sp_list[ind],
             'se': self.se_list[ind]})
        return df_ost_2

    def _generate_interception(self, ind):
        #        sp_list = []
        #        for i in range(len(self.sp_list)):
        #            sp_list.append(1-self.sp_list[i])
        #       df_ost_2 = pd.DataFrame(
        #           {'dx': self.dx_list, 'tp': self.tp_list, 'fp': self.fp_list, 'fn': self.fn_list, 'tn': self.tn_list,
        #            'sp': sp_list, 'se': self.se_list})
        fig_rasp_2 = go.Figure()
        # px.scatter(df_ost_2, x="dx", y="se")
        fig_rasp_2.add_trace(
            go.Scatter(
                x=self.dx_list[ind],
                y=self.sp_list[ind],
                mode="lines+markers",
                #             xaxis='dx',
                #             yaxis='sp',
                line=go.scatter.Line(color="red"),
                showlegend=True)
        )

        fig_rasp_2.add_trace(
            go.Scatter(
                x=self.dx_list[ind],
                y=self.se_list[ind],
                #            xaxis='dx',
                #            yaxis='se',
                mode="lines+markers",
                line=go.scatter.Line(color="gray"),
                showlegend=True)
        )

        fig_rasp_2.update_xaxes(
            title_text="Порог отсечения",
            title_font={"size": 20},
            title_standoff=25)
        fig_rasp_2.update_yaxes(
            title_text="Значение",
            title_font={"size": 20},
            title_standoff=25)
        fig_rasp_2.update_traces(marker_size=10)
        # fig_rasp_2.add_trace(px.scatter(df_ost_2, x="dx", y="se"))
        fig_rasp_2.update_traces(marker_size=10)

        return fig_rasp_2

    def _generate_inter_table(self, ind):
        #        sp_list = []
        #        for i in range(len(self.sp_list)):
        #            sp_list.append(1-self.sp_list[i])
        df_ost_2 = pd.DataFrame(
            {'sp': self.sp_list[ind], 'se': self.se_list[ind]})
        return df_ost_2

    def _generate_dashboard(self):
        # точки для разных ROC
        columns_list = self.predict.df_X.columns
        y_true = self.predict.df_Y
        for i in range(len(columns_list)):
            df_X = self.predict.df_X[columns_list[i]]
            dx = (max(df_X) - min(df_X)) / (len(df_X) - 1)
            dx_init = 0  # min(df_X) - 0.05 * dx
            y_pred = y_true.copy(deep=True)

            dx_list = []
            tp_list = []
            tn_list = []
            fp_list = []
            fn_list = []
            se_list = []
            sp_list = []
            inv_sp_list = []

            flag = True
            while True:
                if flag:
                    dx_init = min(df_X) - 0.05 * dx
                    flag = False
                else:
                    dx_init += dx

                for j in range(len(y_true)):
                    if df_X[j] < dx_init:
                        y_pred[j] = 0
                    else:
                        y_pred[j] = 1

                TN, FP, FN, TP = sm.confusion_matrix(y_true, y_pred).ravel()
                se = TP / (TP + FN)
                sp = TN / (TN + FP)

                if (len(tp_list) == 0) or (
                        len(tp_list) > 0 and (TP != tp_list[-1] or TN != tn_list[-1] or FP != fp_list[-1])):
                    dx_list.append(round(dx_init, 3))
                    tp_list.append(TP)
                    tn_list.append(TN)
                    fp_list.append(FP)
                    fn_list.append(FN)
                    se_list.append(round(se, 3))
                    sp_list.append(round(sp, 3))
                    inv_sp_list.append(round((1 - sp), 3))

                if not dx_init < max(df_X):
                    break

            self.dx_list.append(dx_list)
            self.tp_list.append(tp_list)
            self.tn_list.append(tn_list)
            self.fp_list.append(fp_list)
            self.fn_list.append(fn_list)
            self.sp_list.append(sp_list)
            self.se_list.append(se_list)
            self.inv_sp_list.append(inv_sp_list)

        # сама таблица точек
        df_dots = self._generate_dots(0)

        # таблица метрик
        df_metrics = self._generate_metrics(0)

        metric_list = self.predict.settings['metrics']
        for item in reversed(df_metrics.columns.tolist()):
            if item == 'Threshold' and 'trashhold' not in metric_list:
                df_metrics.pop(item)
            if item == 'Оптимальный порог' and 'trashhold' not in metric_list:
                df_metrics.pop(item)
            if item == 'Accuracy' and 'accuracy' not in metric_list:
                df_metrics.pop(item)
            if item == 'Точность' and 'precision' not in metric_list:
                df_metrics.pop(item)
            if item == 'F-мера' and 'F' not in metric_list:
                df_metrics.pop(item)
            if item == 'Полнота' and 'recall' not in metric_list:
                df_metrics.pop(item)
            if item == 'Доверительный интервал' and 'confidence' not in metric_list:
                df_metrics.pop(item)

        # ROC-кривая
        fig_roc = go.Figure()

        # график пересечения
        fig_inter = go.Figure()

        # точки для графика пересечения
        df_inter = self._generate_inter_table(0)

        def update_roc(column_name, self=self):
            fig_roc = go.Figure()
            ind = columns_list.tolist().index(column_name)
            dov_int = (np.var(self.se_list[
                       ind]) / (len(self.se_list[ind]) * (len(self.se_list[ind]) - 1))) ** 0.5
            dov_list_1 = [self.se_list[ind][i] - 1.96 *
                          dov_int for i in range(len(self.se_list[ind]))]
            dov_list_2 = [self.se_list[ind][i] + 1.96 *
                          dov_int for i in range(len(self.se_list[ind]))]
            fig_roc.add_trace(
                go.Scatter(
                    x=self.inv_sp_list[ind],
                    y=self.se_list[ind],
                    mode="lines+markers",
                    line=go.scatter.Line(color="red"),
                    fill='tozeroy',
                    name='ROC-кривая',
                    showlegend=True)
            )
            fig_roc.add_trace(
                go.Scatter(
                    x=self.inv_sp_list[ind],
                    y=self.inv_sp_list[ind],
                    mode="lines",
                    line=go.scatter.Line(color="blue"),
                    # fill='tozeroy',
                    showlegend=False)
            )
            fig_roc.add_trace(
                go.Scatter(
                    x=self.inv_sp_list[ind],
                    y=dov_list_1,
                    mode="lines",
                    line=go.scatter.Line(color="gray"),
                    name='Доверительный интервал ROC-кривой',
                    showlegend=True)
            )
            fig_roc.add_trace(
                go.Scatter(
                    x=self.inv_sp_list[ind],
                    y=dov_list_2,
                    mode="lines",
                    line=go.scatter.Line(color="gray"),
                    showlegend=False)
            )

            fig_roc.update_xaxes(
                title_text="1-Специфичность",
                title_font={"size": 20},
                title_standoff=25)
            fig_roc.update_yaxes(
                title_text="Чувствительность",
                title_font={"size": 20},
                title_standoff=25)
            fig_roc.update_traces(marker_size=10)

            return fig_roc

        self.predict.app.callback(dash.dependencies.Output('graph_roc', 'figure'),
                                  dash.dependencies.Input('metric_name', 'value'))(update_roc)

        def update_inter(column_name, self=self):
            fig_inter = go.Figure()
            ind = columns_list.tolist().index(column_name)
            # df = pd.DataFrame({'dx': self.dx_list[ind], 'sp': self.sp_list[ind], 'se': self.se_list[ind]})
            fig_inter.add_trace(
                go.Scatter(
                    x=self.dx_list[ind],
                    y=self.sp_list[ind],
                    mode="lines+markers",
                    line=go.scatter.Line(color="red"),
                    name='Специфичность',
                    # fill='tozeroy',
                    showlegend=True)
            )
            fig_inter.add_trace(
                go.Scatter(
                    x=self.dx_list[ind],
                    y=self.se_list[ind],
                    mode="lines+markers",
                    line=go.scatter.Line(color="blue"),
                    name='Чувствительность',
                    # fill='tozeroy',
                    showlegend=True)
            )
            fig_inter.update_xaxes(
                title_text="Порог отсечения",
                title_font={"size": 20},
                title_standoff=25)
            fig_inter.update_yaxes(
                title_text="Значения",
                title_font={"size": 20},
                title_standoff=25)
            fig_inter.update_traces(marker_size=10)
            return fig_inter

        self.predict.app.callback(dash.dependencies.Output('graph_inter', 'figure'),
                                  dash.dependencies.Input('metric_name', 'value'))(update_inter)

        def update_table_dot(column_name, self=self):
            ind = columns_list.tolist().index(column_name)
            df = pd.DataFrame(
                {'dx': self.dx_list[ind], 'tp': self.tp_list[ind], 'fp': self.fp_list[ind],
                 'fn': self.fn_list[ind], 'tn': self.tn_list[ind],
                 'sp': self.sp_list[ind], 'se': self.se_list[ind]})

            return df.to_dict('records')

        self.predict.app.callback(dash.dependencies.Output('table_dot', 'data'),
                                  dash.dependencies.Input('metric_name', 'value'))(update_table_dot)

        def update_table_inter(column_name, self=self):
            ind = columns_list.tolist().index(column_name)
            df = self._generate_inter_table(ind)

            return df.to_dict('records')

        self.predict.app.callback(dash.dependencies.Output('table_inter', 'data'),
                                  dash.dependencies.Input('metric_name', 'value'))(update_table_inter)

        def update_metrics(column_name):
            ind = columns_list.tolist().index(column_name)
            df = self._generate_metrics(ind)
            return df.to_dict('records')

        self.predict.app.callback(dash.dependencies.Output('table_metrics', 'data'),
                                  dash.dependencies.Input('metric_name', 'value'))(update_metrics)

        div_markdown = html.Div([
                dcc.Markdown(children="Выберите группирующую переменную:"),
                dcc.Dropdown(
                    id='metric_name',
                    options=[{'label': i, 'value': i} for i in columns_list],
                    value=columns_list[0]
                )
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '5px'})

        div_roc = html.Div([
                html.Div(html.H4(children='ROC'), style={
                         'text-align': 'center'}),
                html.Div([
                    html.Div(dcc.Graph(id='graph_roc', figure=fig_roc),
                             style={'text-align': 'center', 'width': '78%', 'display': 'inline-block',
                                    'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid'}),
                    html.Div(dcc.Markdown(roc_roc))])
            ], style={'margin': '50px'})

        div_metrics = html.Div([
                html.Div(html.H4(children='Таблица метрик'),
                         style={'text-align': 'center'}),
                html.Div([
                    html.Div(dash_table.DataTable(
                        id='table_metrics',
                        columns=[{"name": i, "id": i}
                                 for i in df_metrics.columns],
                        data=df_metrics.to_dict('records'),
                        export_format='csv'
                    ), style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'text-align': 'center',
                              'width': str(len(df_metrics.columns) * 10 - 10) + '%', 'display': 'inline-block'}),
                    html.Div(dcc.Markdown(roc_table_metrics))])
            ], style={'margin': '50px'})

        div_dot = html.Div([
                html.Div(html.H4(children='Таблица точек ROC'),
                         style={'text-align': 'center'}),
                html.Div([
                    html.Div(dash_table.DataTable(
                        id='table_dot',
                        columns=[{"name": i, "id": i}
                                 for i in df_dots.columns],
                        data=df_dots.to_dict('records'),
                        export_format='csv'
                    ), style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'text-align': 'center',
                              'width': str(len(df_dots.columns) * 10 - 10) + '%', 'display': 'inline-block'}),
                    html.Div(dcc.Markdown(roc_table))])
            ], style={'margin': '50px'})

        div_inter = html.Div([
                html.Div(html.H4(children='График пересечения'),
                         style={'text-align': 'center'}),
                html.Div([
                    html.Div(dcc.Graph(id='graph_inter', figure=fig_inter),
                             style={'text-align': 'center', 'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid'}),
                    html.Div(dcc.Markdown(roc_inter_graph))])
            ], style={'margin': '50px'})

        div_list = [div_markdown, div_roc, div_metrics, div_dot, div_inter]

        if len(df_metrics.columns.tolist()) == 2 or 'metrics_table' not in metric_list:
            div_list.remove(div_metrics)
        if 'spec_and_sens_table' not in metric_list:
            div_list.remove(div_inter)
        if 'points_table' not in metric_list:
            div_list.remove(div_dot)

            # html.Div([
            #    html.Div(html.H4(children='Точки для графика'), style={'text-align': 'center'}),
            #    html.Div(dash_table.DataTable(
            #        id='table_inter',
            #        columns=[{"name": i, "id": i} for i in df_inter.columns],
            #        data=df_inter.to_dict('records')
            #    ), style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'text-align': 'center',
            #              'width': str(len(df_inter.columns) * 10 - 10) + '%', 'display': 'inline-block'})
            #], style={'margin': '50px'}),
        #, style={'margin': '50px'})

        return html.Div(div_list, style={'margin': '50px'})

    def _generate_comparison(self):
        columns_list = self.predict.df_X.columns
        fig_roc_2 = go.Figure()

        sum_table = pd.DataFrame(
            columns=['Параметр', 'Threshold', 'Оптимальный порог', 'Полнота', 'Точность',
                     'Accuracy', 'F-мера', 'Доверительный интервал', 'AUC'])

        for i in range(len(columns_list)):
            temp_df = self._generate_metrics(i)
            sum_table = pd.concat([sum_table, temp_df], ignore_index=True)
        sum_table.rename(
            columns={'Параметр': 'Группирующая переменная'}, inplace=True)
        sum_table['Группирующая переменная'] = [item for item in columns_list]

        def update_roc_2(param_1, param_2, self=self):
            fig_roc_2 = go.Figure()
            ind_1 = columns_list.tolist().index(param_1)
            ind_2 = columns_list.tolist().index(param_2)
#            dov_int = (np.var(self.se_list[ind]) / (len(self.se_list[ind]) * (len(self.se_list[ind]) - 1))) ** 0.5
#            dov_list_1 = [self.se_list[ind][i] - 1.96 * dov_int for i in range(len(self.se_list[ind]))]
#            dov_list_2 = [self.se_list[ind][i] + 1.96 * dov_int for i in range(len(self.se_list[ind]))]
            fig_roc_2.add_trace(
                go.Scatter(
                    x=self.inv_sp_list[ind_1],
                    y=self.se_list[ind_1],
                    mode="lines+markers",
                    line=go.scatter.Line(color="red"),
                    fill='tozeroy',
                    name=param_1,
                    showlegend=True)
            )
            fig_roc_2.add_trace(
                go.Scatter(
                    x=self.inv_sp_list[ind_1],
                    y=self.inv_sp_list[ind_1],
                    mode="lines",
                    line=go.scatter.Line(color="green"),
                    # fill='tozeroy',
                    showlegend=False)
            )
            fig_roc_2.add_trace(
                go.Scatter(
                    x=self.inv_sp_list[ind_2],
                    y=self.se_list[ind_2],
                    mode="lines+markers",
                    line=go.scatter.Line(color="blue"),
                    fill='tozeroy',
                    name=param_2,
                    showlegend=True)
            )

            fig_roc_2.update_xaxes(
                title_text="1-Специфичность",
                title_font={"size": 20},
                title_standoff=25)
            fig_roc_2.update_yaxes(
                title_text="Чувствительность",
                title_font={"size": 20},
                title_standoff=25)
            fig_roc_2.update_traces(marker_size=10)

            return fig_roc_2

        self.predict.app.callback(dash.dependencies.Output('graph_roc_2', 'figure'),
                                  [dash.dependencies.Input('group_param_1', 'value'),
                                   dash.dependencies.Input('group_param_2', 'value')])(update_roc_2)

        div_2_title = html.Div(html.H2(children='Блок сравнения'), style={'text-align': 'center'})

        div_2_markdown = html.Div([
                html.Div([
                    dcc.Markdown(
                        children="Выберите первую группирующую переменную:"),
                    dcc.Dropdown(
                        id='group_param_1',
                        options=[{'label': i, 'value': i}
                                 for i in columns_list],
                        value=columns_list[0]
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Markdown(
                        children="Выберите вторую группирующую переменную:"),
                    dcc.Dropdown(
                        id='group_param_2',
                        options=[{'label': i, 'value': i}
                                 for i in columns_list],
                        value=columns_list[1]
                    )
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ], style={'padding': '5px'})

        div_2_roc = html.Div([
                html.Div(html.H4(children='ROC'), style={
                         'text-align': 'center'}),
                html.Div([
                    html.Div(dcc.Graph(id='graph_roc_2', figure=fig_roc_2),
                             style={'text-align': 'center', 'width': '78%', 'display': 'inline-block',
                                    'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid'}),
                    html.Div(dcc.Markdown(roc_comp_roc))])
            ], style={'margin': '50px'})

        div_2_metrics = html.Div([
                html.Div(html.H4(children='Таблица метрик'),
                         style={'text-align': 'center'}),
                html.Div([
                    html.Div(dash_table.DataTable(
                        id='table_metrics_2',
                        columns=[{"name": i, "id": i}
                                 for i in sum_table.columns],
                        data=sum_table.to_dict('records'),
                        export_format='csv'
                    ), style={'border-color': 'rgb(220, 220, 220)', 'border-style': 'solid', 'text-align': 'center',
                              'width': str(len(sum_table.columns) * 10 - 10) + '%', 'display': 'inline-block'}),
                    html.Div(dcc.Markdown(roc_comp_metrics))])
            ], style={'margin': '50px'})

        div_2_list = [div_2_title, div_2_markdown, div_2_roc, div_2_metrics]

        return html.Div(div_2_list, style={'margin': '50px'})
