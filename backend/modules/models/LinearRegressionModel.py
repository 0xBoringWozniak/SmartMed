import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn.metrics as sm
import pandas as pd
from scipy import stats
from ModelInterface import Model
import linear_text

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re


class ModelController():
    def __init__(self, setting):
        if settings['model'] == 'linreg':
            self.model = LinearRegressionModel()
        else:
            pass


class LinearRegressionModel(ModelController):

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, x: np.array, y: np.array) -> np.array:
        self.model = self.model.fit(x, y)

    def score(self, x: np.array, y: np.array) -> float:  # точность модели (R^2)
        return self.model.score(x, y)

    def predict(self, x: np.array) -> float:  # предсказанное значение для числа или списка
        return self.model.predict(x)

    def get_resid(self) -> np.array:
        return self.model.coef_

    def get_intercept(self):  # коэффициент пересечения
        return self.model.intercept_

    def get_all_coef(self):  # коэффициенты с пересечением
        return np.append(self.model.intercept_, self.model.coef_)

    def make_X(self, def_df, def_names):  # создаёт датафрейм признаков
        df1 = pd.DataFrame()
        for name in def_names:
            df1 = pd.concat([df1, def_df[name]], axis=1)
        return df1

    def make_Y(self, def_df, def_name):  # создаёт массив зависимой переменной
        return def_df[def_name]

    def get_mean(self, def_df_Y):  # среднее значение Y
        return sum(def_df_Y) / len(def_df_Y)

    def get_TSS(self, def_df_Y, def_mean_Y):  # дисперсия Y
        def_TSS = 0
        for i in range(len(def_df_Y)):
            def_TSS += (def_df_Y[i] - def_mean_Y) ** 2
        return def_TSS

    def get_RSS(self, def_predict_Y, def_mean_Y):  # доля объяснённой дисперсии
        def_RSS = 0
        for i in range(len(def_predict_Y)):
            def_RSS += (def_predict_Y[i] - def_mean_Y) ** 2
        return def_RSS

    def get_ESS(self, def_df_Y, def_predict_Y):  # доля необъяснённой дисперсии
        def_ESS = 0
        for i in range(len(def_df_Y)):
            def_ESS += (def_df_Y[i] - def_predict_Y[i]) ** 2
        return def_ESS

    def get_R(self, def_df_Y, def_predict_Y):  # коэффицент множественной корреляции
        return sm.r2_score(def_df_Y, def_predict_Y) ** 0.5

    def get_deg_fr(self, def_df_X):  # степени свободы в списке
        k1 = def_df_X.shape[1]
        k2 = def_df_X.shape[0] - def_df_X.shape[1] - 1
        return [k1, k2]

    def get_st_err(self, def_RSS, def_de_fr):  # стандартная ошибка оценки уравнения
        return (def_RSS / (def_de_fr[1] - 2)) ** 0.5

    def get_cov_matrix(self, def_df_X):  # обратная ковариационная матрица
        df2_X = def_df_X.copy()
        df2_X.insert(0, '1', np.ones((df2_X.shape[0], 1)))
        df2_X_T = df2_X.values.transpose()
        return np.linalg.inv(np.dot(df2_X_T, df2_X))

    def get_cov_matrix_2(self, df_X):  # обратная ковариационная матрица для расстояний Махалонобиса
        df2_X = df_X.copy()
        df2_X_T = df2_X.values.transpose()
        return np.linalg.inv(np.dot(df2_X_T, df2_X))

    def uravnenie(self, def_b, def_names, def_name):  # уравнение регрессии
        def_st = 'Y=' + str(round(def_b[0], 3))
        for i in range(1, len(def_b)):
            if def_b[i] > 0:
                def_st += ' + ' + str(round(def_b[i], 3)) + 'X(' + str(i) + ')'
            else:
                def_st += ' - ' + str(round(abs(def_b[i]), 3)) + 'X(' + str(i) + ')'
        def_st += ', где:'  # \nX(0)-константа'
        uravlist = [def_st]
        uravlist.append('\n')
        uravlist.append('Y - ' + def_name + ';')
        for i in range(1, len(def_b)):
            uravlist.append('\n')
            uravlist.append('X(' + str(i) + ') - ' + def_names[i - 1] + ';')
        return uravlist

    def st_coef(self, def_df_X, def_TSS, b):  # стандартизованнные коэффициенты
        def_b = list(b)
        def_b.pop(0)
        b_st = []
        for i in range(len(def_b)):
            a = def_df_X.iloc[:, i]
            mean_X = cntrl.model.get_mean(a)
            sx = cntrl.model.get_TSS(a, mean_X)
            b_st.append(def_b[i] * (sx / def_TSS) ** 0.5)
        return b_st

    def st_er_coef(self, def_df_Y, def_predict_Y, def_cov_mat):  # стандартные ошибки
        def_MSE = np.mean((def_df_Y - def_predict_Y.T) ** 2)
        var_est = def_MSE * np.diag(def_cov_mat)
        SE_est = np.sqrt(var_est)
        return SE_est

    def t_stat(self, def_df_X, def_df_Y, def_predict_Y, def_d_free, def_b):  # t-критерии коэффициентов
        s = np.sum((def_predict_Y - def_df_Y) ** 2) / (def_d_free[1] + 1)
        df2_X = def_df_X.copy()
        df2_X.insert(0, '1', np.ones((df2_X.shape[0], 1)))
        sd = np.sqrt(s * (np.diag(np.linalg.pinv(np.dot(df2_X.T, df2_X)))))
        def_t_stat = []
        for i in range(len(def_b)):
            def_t_stat.append(def_b[i] / sd[i])
        return def_t_stat

    def get_RMSD(self, def_df_Y, def_predict_Y):  # корень из среднеквадратичной ошибки
        return np.sqrt(sm.mean_squared_error(def_df_Y, def_predict_Y))

    def get_MSE(self, def_df_Y, def_predict_Y):  # среднеквадратичная ошибка
        return sm.mean_squared_error(def_df_Y, def_predict_Y)

    def get_MAE(self, def_df_Y, def_predict_Y):  # средняя абсолютная ошибка
        return sm.mean_absolute_error(def_df_Y, def_predict_Y)

    def get_R2_adj(self, def_df_X, def_df_Y, def_predict_Y):  # R^2 adjusted
        return 1 - (1 - sm.r2_score(def_df_Y, def_predict_Y)) * (
                    (len(def_df_X) - 1) / (len(def_df_X) - def_df_X.shape[1] - 1))

    def get_Fst(self, def_df_X, def_df_Y, def_predict_Y):  # F-статистика
        r2 = sm.r2_score(def_df_Y, def_predict_Y)
        return r2 / (1 - r2) * (len(def_df_X) - def_df_X.shape[1] - 1) / def_df_X.shape[1]

    def p_values(self, def_df_X, def_t_stat):
        newX = pd.DataFrame({"Constant": np.ones(def_df_X.shape[0])}).join(def_df_X)
        p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - len(newX.columns) - 1))) for i in def_t_stat]
        return p_values


# должно передаваться: settings, names, name_Y, df_0

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# это должно передаваться, но пока так
settings = {'model': 'linreg'}
df_0 = pd.read_csv('C:/winequality-red.csv', sep=';')
names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
         'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
name_Y = 'quality'

# создание модели
cntrl = ModelController(settings)  # обязательно
df_X = cntrl.model.make_X(df_0, names)  # обязательно
df_Y = cntrl.model.make_Y(df_0, name_Y)  # обязательно
cntrl.model.fit(df_X, df_Y)  # обязательно
b = cntrl.model.get_all_coef()
mean_Y = cntrl.model.get_mean(df_Y)
TSS = cntrl.model.get_TSS(df_Y, mean_Y)
de_fr = cntrl.model.get_deg_fr(df_X)
b_st = cntrl.model.st_coef(df_X, TSS, b)
predict_Y = cntrl.model.predict(df_X)  # обязательно #рассчитанные Y
RSS = cntrl.model.get_RSS(predict_Y, mean_Y)
ESS = cntrl.model.get_ESS(df_Y, predict_Y)

# Таблица с критериями качества модели
df_result_1 = pd.DataFrame(columns=['Параметр', 'Значение'])
result_1 = ['R', 'R2', 'R2adj', 'df', 'Fst', 'StError']  # список значений, которые требуется рассчитать
i = 1
for name in result_1:
    if name == 'R':
        df_result_1.loc[i] = ['R', round(cntrl.model.get_R(df_Y, predict_Y), 3)]
        i += 1
    if name == 'R2':
        df_result_1.loc[i] = ['R2', round(cntrl.model.score(df_X, df_Y), 3)]
        i += 1
    if name == 'R2adj':
        df_result_1.loc[i] = ['R2adj', round(cntrl.model.get_R2_adj(df_X, df_Y, predict_Y), 3)]
        i += 1
    if name == 'df':
        df_result_1.loc[i] = ['df',
                              str(str(cntrl.model.get_deg_fr(df_X)[0]) + '; ' + str(cntrl.model.get_deg_fr(df_X)[1]))]
        i += 1
    if name == 'Fst':
        df_result_1.loc[i] = ['Fst', round(cntrl.model.get_Fst(df_X, df_Y, predict_Y), 3)]
        i += 1
    if name == 'StError':
        df_result_1.loc[i] = ['St.Error', round(cntrl.model.get_st_err(RSS, de_fr), 3)]
        i += 1

# Таблица с критериями признаков
df_column = list(df_X.columns)
df_column.insert(0, 'Параметр')
df_result_2 = pd.DataFrame(columns=df_column)
result_2 = ['b', 'bst', 'StErrorb', 'Tst', 'pvalue']  # список значений, которые требуется рассчитать
i = 1
t_st = cntrl.model.t_stat(df_X, df_Y, predict_Y, de_fr, b)
cov_mat = cntrl.model.get_cov_matrix(df_X)
st_er_coef = cntrl.model.st_er_coef(df_Y, predict_Y, cov_mat)
p_values = cntrl.model.p_values(df_X, t_st)
for name in result_2:
    if name == 'b':
        res_b = ['b']
        list_b = list(b)
        for j in range(1, len(list_b)):
            res_b.append(round(list_b[j], 3))
        df_result_2.loc[i] = res_b
        i += 1
    if name == 'bst':
        res_bst = ['b_st']
        list_bst = list(b_st)
        for j in range(len(list_bst)):
            res_bst.append(round(list_bst[j], 3))
        df_result_2.loc[i] = res_bst
        i += 1
    if name == 'StErrorb':
        res_errb = ['St.Error b']
        st_er_b = list(st_er_coef)
        for j in range(1, len(st_er_b)):
            res_errb.append(round(st_er_b[j], 3))
        df_result_2.loc[i] = res_errb
        i += 1
    if name == 'Tst':
        res_tst = ['t-критерий']
        for j in range(1, len(t_st)):
            res_tst.append(round(t_st[j], 3))
        df_result_2.loc[i] = res_tst
        i += 1
    if name == 'pvalue':
        res_pval = ['p-value']
        for j in range(1, len(t_st)):
            res_pval.append(round(p_values[j], 3))
        df_result_2.loc[i] = res_pval
        i += 1

# Вывод уравнения
uravnenie = cntrl.model.uravnenie(b, names, name_Y)

# Таблица остатков
d_1 = df_Y  # зависимый признак
d_2 = predict_Y  # предсказанное значение
d_3 = df_Y - predict_Y  # остатки
d_4 = (predict_Y - mean_Y) / ((TSS / len(predict_Y)) ** 0.5)  # стандартизированные предсказанные значения
d_5 = (df_Y - predict_Y) / ((ESS / len(df_Y)) ** 0.5)
d_6 = df_Y * 0 + ((cntrl.model.get_st_err(RSS, de_fr) / len(df_Y)) ** 0.5)

mean_list = []  # средние значения для каждого признака
for i in range(df_X.shape[1]):
    a = df_X.iloc[:, i]
    mean_list.append(cntrl.model.get_mean(a))

mah_df = []  # тут будут расстояния Махалонобиса для всех наблюдений
cov_mat_2 = cntrl.model.get_cov_matrix_2(df_X)  # ков. матрица без единичного столбца
for j in range(df_X.shape[0]):
    aa = df_X.iloc[j, :]  # строка с признаками
    meann = []  # список отличий от среднего
    for i in range(df_X.shape[1]):
        meann.append(mean_list[i] - aa[i])
    mah_df.append(np.dot(np.dot(np.transpose(meann), cov_mat_2), meann))  # расстояние для наблюдения

df_result_3 = pd.DataFrame({'Номер наблюдения': 0, 'Исходное значение признака': np.round(d_1, 3),
                            'Рассчитанное значение признака': np.round(d_2, 3), 'Остатки': np.round(d_3, 3),
                            'Стандартные предсказанные значения': np.round(d_4, 3),
                            'Стандартизированные остатки': np.round(d_5, 3),
                            'Стандартная ошибка предсказанного значения': np.round(d_6, 3),
                            'Расстояние Махаланобиса': np.round(mah_df, 3)})
df_result_3.iloc[:, 0] = [i + 1 for i in range(df_result_3.shape[0])]

# график остатков
number = 0  # это для функции
a = [i + 1 for i in range(len(df_Y))]
st0 = go.Figure()
st0.add_trace(go.Scatter(x=a, y=df_Y - predict_Y, mode='markers', name='initial_values'))
st0.update_traces(marker_size=20)

# График распределения остатков
fig_rasp = go.Figure()
df_ost = pd.DataFrame({'Изначальный Y': df_Y, 'Предсказанный Y': predict_Y})
fig_rasp = px.scatter(df_ost, x="Изначальный Y", y="Предсказанный Y", trendline="ols")
fig_rasp.update_traces(marker_size=20)

fig_rasp_1 = go.Figure()
df_ost_1 = pd.DataFrame({'Изначальный Y': d_1, 'Ст остатки': d_5})
fig_rasp_1 = px.scatter(df_ost_1, x="Изначальный Y", y="Ст остатки", trendline="ols")
fig_rasp_1.update_traces(marker_size=20)

fig_rasp_2 = go.Figure()
df_ost_2 = pd.DataFrame({'Номер': 0, 'Предсказанный Y': predict_Y})
df_ost_2.iloc[:, 0] = [i + 1 for i in range(df_ost_2.shape[0])]
fig_rasp_2 = px.scatter(df_ost_2, x="Номер", y="Предсказанный Y", trendline="ols")
fig_rasp_2.update_traces(marker_size=20)
app.layout = html.Div([
    html.Div(html.H1(children='Линейная регрессия'), style={'text-align': 'center'}),
    html.Div(html.H2(children='Уравнение множественной регрессии'), style={'text-align': 'center'}),
    dcc.Markdown(id='Markdown', children=uravnenie),
    html.Div(html.H2(children='Критерии качества модели'), style={'text-align': 'center'}),
    html.Div(dash_table.DataTable(
        id='table1',
        columns=[{"name": i, "id": i} for i in df_result_1.columns],
        data=df_result_1.to_dict('records')
    ), style={'width': str(len(df_result_1.columns) * 5 - 10) + '%', 'display': 'inline-block'}),
    html.Div(dcc.Markdown(linear_text.markdown_linear_table1)),
    html.Div(html.H2(children='Критерии значимости переменных'), style={'text-align': 'center'}),
    html.Div(dash_table.DataTable(
        id='table2',
        columns=[{"name": i, "id": i} for i in df_result_2.columns],
        data=df_result_2.to_dict('records'),
        style_cell={
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'maxWidth': 0,  # len(df_result_3.columns)*5,
        },
    ), style={'width': str(len(df_result_2.columns) * 5 - 10) + '%', 'display': 'inline-block'}),
    html.Div(dcc.Markdown(linear_text.markdown_linear_table2)),
    html.Div(html.H2(children='Таблица остатков'), style={'text-align': 'center'}),
    html.Div(dash_table.DataTable(
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
    html.Div(dcc.Markdown(linear_text.markdown_linear_table3)),
    html.Div([html.Div(html.H2(children='График 1'), style={'text-align': 'center'}),
              dcc.Graph(id='Graph_ost_1', figure=st0)]),
    html.Div([html.Div(html.H2(children='График 2'), style={'text-align': 'center'}),
              dcc.Graph(id='Graph_ost_2', figure=fig_rasp)]),
    dcc.Input(id='input-1-state', type='text', value=''),
    html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
    html.Div(id='output-state'),
    html.Div([html.Div(html.H2(children='График 4'), style={'text-align': 'center'}),
              dcc.Graph(id='Graph_ost_4', figure=fig_rasp_2)]),
])
coord_list = []  # массив координат для новых точек на графике


@app.callback(Output('output-state', 'children'),
              [Input('submit-button-state', 'n_clicks')],
              [State('input-1-state', 'value')])
def update_output(n_clicks, input1):
    # global number
    global coord_list
    global b
    number = len(coord_list)
    if n_clicks == 0 or input1 == 'Да':
        coord_list = []
        number = len(coord_list)
        return u'''{} Введите значение параметра "{}"'''.format(number, df_X.columns[0])
    if re.fullmatch(r'^([-+])?\d+([,.]\d+)?$', input1):
        number += 1
        if input1.find(',') > 0:
            input1 = float(input1[0:input1.find(',')] + '.' + input1[input1.find(',') + 1:len(input1)])
        coord_list.append(float(input1))
        if len(coord_list) < len(df_X.columns):
            return u'''{} Введите значение параметра  "{}"'''.format(number, df_X.columns[
                number])  # максимальное значнеие - len(df_X.columns)-1
        if len(coord_list) == len(df_X.columns):
            number = -1
            yzn = b[0]
            for i in range(len(coord_list)):
                yzn += coord_list[i] * b[i + 1]
            return u'''Предсказанное значение равно {} \n Если желаете посчитать ещё для одного набор признаков, напишите "Да"'''.format(
                round(yzn, 3))
    elif n_clicks > 0:
        return u'''{} Введено не число, введите значение параметра "{}" повторно'''.format(number, df_X.columns[number])
    if number == -1 and input1 != 0 and input1 != 'Да' and input1 != '0':
        return u'''Если желаете посчитать ещё для {} набор признаков, напишите "Да"'''.format('одного')

if __name__ == '__main__':
    app.run_server(debug=True)