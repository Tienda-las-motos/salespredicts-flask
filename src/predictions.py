from fbclient import FirebaseApp
from src.file import upload_file, upload_img
from src.products import formatValues

from sklearn.svm import SVR
from pmdarima.arima import auto_arima
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# from IPython.core.display import display
from datetime import datetime

import time
import pytz
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mlp
import numpy as np
import json
import urllib.request
import os

db = FirebaseApp.fs
st = FirebaseApp.st
mlp.style.use('seaborn')
tz = pytz.timezone('America/Mexico_City')


def ValidateRequest(query_params, request):

    print(request)
    global message
    global params
    params = {}
    for param in query_params:
        print(param, request[param])
        try:
            params[param] = request[param]
        except:
            message = f'La petición debe incluir el parámetro "{param}"'
            raise

    return params


class SalesPredictions():
    def months_query(request):

        # VALIDATE PARAMS
        try:
            params = ValidateRequest(
                ['table', 'product', 'months'], request.args)
        except:
            return {
                'message': 'Falto un parámetro en la petición',
                'status': 400
            }, 400

        table_id = params['table']
        product_id = params['product']
        query_months = int(params['months'])
        # locale.setlocale(locale.LC_TIME, 'es_MX.UTF-8')

        # IMPORT FILE
        global cloud_path
        global local_path
        cloud_path = 'tables/'+table_id+'/products/'+product_id
        doc_ref = db.document(cloud_path)

        print(cloud_path)
        try:
            doc = doc_ref.get()
        except:
            return {
                'message': 'No se encontró el documento en la base de datos',
                'status': 404
            }
        print('Petición realizada correctamente')

        print(doc.exists)
        # READ TABLE
        if doc.exists:
            doc_URL = doc.to_dict()['time_stats']['files']['meses_list_df']
            try:
                dataset = pd.read_csv(doc_URL,  decimal=".")
            except:
                return {
                    'message': 'Hace falta la lista de ventas por mes',
                    'status': 404
                }, 404
        else:
            return {
                'message': 'No existe el archivo solicitado',
                'status': 404
            }, 404

        local_path = os.path.abspath(os.path.dirname(__file__))+'/'
        print('Archivo cargado')
        # display(dataset.head())

        # TRAIN AND PREDICT
        year_dataset, reg, score, predictionsURL = Predictions.by_year_sales(
            dataset)

        print('trained')

        # meses que el usuario quiere conocer en predicción
        months_query = int(query_months)
        months_required = score[1:months_query + 1]

        predict_months = reg.predict(months_required)
        predicted_cant = math.ceil(predict_months.sum())
        print('Se venderán', predicted_cant,
              'Unidades en', months_query, 'meses')

        # plt.plot(predict_year, 'ro', predict_months, 'bo')
        # plt.savefig(local_path+'months_prediction.jpg')
        # monthpredictionURL = upload_file(local_path, cloud_path, 'months_prediction.jpg')

        result = {
            "predicted_cant": int(predicted_cant),
            # "months_prediction_chart": monthpredictionURL,
        }

        doc_ref.update({"year_predictions_URL": predictionsURL})

        return {
            "result": result,
            "status": 200,
            "message": "ok",
        }, 200

    def cant_query(request):
        # locale.setlocale(locale.LC_TIME, 'es_MX.UTF-8')

        try:
            params = ValidateRequest(
                ['table', 'product', 'cant'], request.args)
        except:
            return {
                'message': 'Falto un parámetro en la petición',
                'status': 400
            }, 400

        table_id = params['table']
        product_id = params['product']
        query_cant = params['cant']

        # IMPORT FILE
        global cloud_path
        cloud_path = 'tables/'+table_id+'/products/'+product_id
        doc_ref = db.document(cloud_path)

        try:
            doc = doc_ref.get()
        except:
            return {
                'message': 'No se encontró el documento en la base de datos',
                'status': 404
            }, 404

        print('Petición realizada correctamente')

        global local_path
        doc_URL = doc.to_dict()['time_stats']['files']['meses_list_df']
        dataset = pd.read_csv(doc_URL,  decimal=".")
        local_path = os.path.abspath(os.path.dirname(__file__))+'/'

        print('Archivo cargado')
        # display(dataset.head())

        # TRAIN AND PREDICT
        year_dataset, reg, score, predictionsURL = Predictions.by_year_sales(
            dataset)
        print('trained')

        remaining_cant = float(query_cant)
        months_required = score[0:12]
        predict_months = reg.predict(months_required)
        print(predict_months)
        months_cant = 0
        for cant in predict_months:
            remaining_cant -= cant
            if remaining_cant > 0:
                months_cant = months_cant + 1
            else:
                break

        print(f"{query_cant} Unidades se venderán en {months_cant + 1} meses")

        result = {
            "months_cant": float(months_cant) + 1,
        }

        doc_ref.update({"year_predictions_URL": predictionsURL})
        return {
            "result": result,
            "status": 200,
            "message": "ok",
        }, 200

    def stats_query(request):
        # VALIDATE THERE IS TABLE ID

        try:
            query = ValidateRequest(
                ['table', 'product', 'test_size', 'window_size'], request.json)
        except:
            return {
                'message': 'Falto un parámetro en la petición',
                'status': 400
            }, 400

        query['test_size'] = int(query['test_size'])
        query['window_size'] = int(query['window_size'])

        # IMPORT FILE
        # locale.setlocale(locale.LC_TIME, 'es_MX.UTF-8')
        global cloud_path
        cloud_path = 'tables/'+query['table']+'/products/'+query['product']
        doc_ref = db.document(cloud_path)

        try:
            doc = doc_ref.get()
        except:
            return {
                'message': 'No se encontró el documento en la base de datos',
                'status': 404
            }, 404
        print('Firestore doc getted')

        try:
            doc_URL = doc.to_dict()['time_stats']['files']['timeline']
        except:
            return {
                "message": "No se encontró el archivo",
                "status": 404
            }, 404
        print('timeline dataset getted')

        product_name = doc.to_dict()['name']
        print(doc_URL)
        dataset = pd.read_csv(doc_URL)
        dataset = dataset.fillna(method='ffill')
        print('dataset defined')

        try:
            predict_results = Predictions.by_stats(
                dataset, query['test_size'], query['window_size'], product_name)
        except Exception as message:
            return {
                'message': message.__str__(),
                'status': 400
            }, 400

        try:
            doc_ref.collection(u'predictions').document(
                u'estimated').set(predict_results)
        except:
            return {
                'message': 'No se pudo guardar',
                'status': 500
            }, 500

        print('si se guardó')

        return {
            "result": predict_results,
            "status": 200,
            "message": "ok",
        }, 200

    def seasonal(request):
        try:
            query = ValidateRequest(
                ['table', 'product', 'test_size'], request.args)
        except:
            return {
                'message': 'Falto un parámetro en la petición',
                'status': 400
            }, 400

        query['test_size'] = int(query['test_size'])

        # locale.setlocale(locale.LC_TIME, 'es_MX.UTF-8')
        cloud_path = 'tables/'+query['table']+'/products/'+query['product']
        doc_ref = db.document(cloud_path)

        try:
            doc = doc_ref.get()
        except:
            return {
                'message': 'No se encontró el documento en la base de datos',
                'status': 404
            }, 404
        # files = doc.to_dict()['files']

        # VALIDATE DATASET
        try:
            doc_URL = doc.to_dict()[
                'time_stats']['files']['unitsbymonths_df_URL']
        except:
            return {
                'message': 'Falta dataset de datos normalizados',
                'status': 404
            }, 404

        print('firebase ok')

        dataset = pd.read_csv(doc_URL, header=None,
                              index_col=0, parse_dates=True, squeeze=True)
        # dataset = formatValues(dataset)

        # split_point = math.floor( len(month_sales) *.80)
        # dataset, validation = month_sales[0:split_point], month_sales[split_point:]
        # print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
        # dataset.to_csv('dataset.csv', header=False)
        # validation.to_csv('validation.csv', header=False)

        print(dataset)
        groups = dataset.groupby(pd.Grouper(freq='M'))
        sells_avg = groups.describe()['count'].sum()/len(groups)

        print('doc readed')
        print(dataset)
        try:
            Arima_model = auto_arima(dataset, start_p=1, start_q=1, max_p=8, max_q=8, start_P=0, start_Q=0, max_P=8, max_Q=8,
                                     m=12, seasonal=True, trace=True, d=1, D=1, error_action='warn', suppress_warnings=True, random_state=20, n_fits=30)
        except:
            return {
                'message': 'No hay suficientes datos para realizar esta predicción',
                'status': 400
            }, 400

        predict = Arima_model.predict(n_periods=query['test_size'])
        total_value_predictions = np.sum(predict)
        months_predicted = total_value_predictions / sells_avg

        print('prediction ok')

        plt.figure(figsize=(12, 6))
        plt.plot(predict)
        arima_pred_imgURL = upload_img(cloud_path, 'arima_prediction.jpg', plt)

        predict_df = pd.DataFrame(data=predict)
        predict_JSON = predict_df.to_json(orient="index")
        arima_pred_jsonURL = upload_file(
            cloud_path, 'arima_prediction.json', predict_JSON)
        print('files ok')

        predict_result = {
            "avg_for_sell": float("{:.2f}".format(sells_avg)),
            "total_predicted": int(total_value_predictions),
            "months_predicted": float("{:.2f}".format(months_predicted)),
            "imgURL": arima_pred_imgURL,
            "jsonURL": arima_pred_jsonURL,
        }

        doc_ref.collection(u'predictions').document(
            u'arima').set(predict_result)

        print('saved ok')
        return {
            'result': predict_result,
            'status': 200,
            'message': 'ok',
        }, 200


class Analyze():
    def providers_offers(request):
        # VALIDATE THERE IS TABLE ID

        try:
            query = ValidateRequest(['table', 'product', 'provider', 'buy_price',
                                    'sale_price',  'condition', 'desc', 'stock'], request.json)
        except:
            return {
                'message': 'Falto un parámetro en la petición',
                'status': 400
            }, 400

        query['buy_price'] = int(query['buy_price'])
        query['sale_price'] = int(query['sale_price'])
        query['condition'] = int(query['condition'])
        query['desc'] = int(query['desc']) / 100
        query['stock'] = int(query['stock'])

        print(query['buy_price'], query['desc'])
        print(type(query['buy_price']), type(query['desc']))

        # IMPORT FILE
        global cloud_path
        global local_path
        global dataset

        cloud_path = 'tables/'+query['table']+'/products/'+query['product']
        doc_ref = db.document(cloud_path)
        local_path = os.path.abspath(os.path.dirname(__file__))+'/'
        print(local_path)

        try:
            doc = doc_ref.get()
        except:
            return {
                'message': 'No se encontró el documento en la base de datos',
                'status': 404
            }, 404

        try:
            doc.to_dict()['year_predictions_URL']
            doc_URL = doc.to_dict()['year_predictions_URL']
            with urllib.request.urlopen(doc_URL) as url:
                dataset = json.loads(url.read())
        except:
            print('se creará el archivo')
            doc_URL = doc.to_dict()['time_stats']['files']['meses_list_df']
            month_sales = pd.read_csv(doc_URL,  decimal=".")
            train_result = Predictions.by_year_sales(month_sales)
            dataset = train_result[0]

        # DEF DATA
        suggest_sale_price = doc.to_dict()['sell_stats']['suggest_sale_price']
        suggest_buy_price = doc.to_dict()['buy_stats']['suggest_buy_price']
        avg_buy_price = doc.to_dict()['product_stats']['avg_buy_price']

        inv_cap = query['stock'] * avg_buy_price
        saving = query['buy_price'] * query['desc']
        desc_price = avg_buy_price - saving
        total_saving = saving * query['condition']
        invest = desc_price * query['condition']
        total_inv = inv_cap + invest
        remaining_inv = -(total_inv)
        remaining_stock = query['stock'] + query['condition']
        year_sales = dataset[0:12]

        global profits
        global utilities
        global porUtilities
        global viability
        global message

        profits = []
        invests = [remaining_inv]
        month1 = 0
        month2 = 0
        print(suggest_sale_price)
        for cant in year_sales:
            # print('cantidad restante', remaining_stock)
            remaining_stock = remaining_stock - cant
            possible_sales = cant * query['sale_price']
            # print('posibles ventas',possible_sales)
            remaining_inv = remaining_inv + possible_sales
            # print('inversión restante',remaining_inv)
            invests.append(int(remaining_inv))
            if remaining_inv > 0:
                if remaining_stock > 0:
                    # print(possible_sales)
                    profits.append(possible_sales)
                    month2 = month2 + 1
            else:
                month1 = month1 + 1
                month2 = month2 + 1

        # print(month1, month2)
        if len(profits) > 0:
            profits = sum(profits)
            # print(profits)
            utilities = profits - total_saving
            # print(utilities)
            if utilities > 0:
                porUtilities = (total_saving * 100)/profits
                viability = True
                message = 'La oferta es conveniente'
            else:
                viability = False
                porUtilities = 0
                message = 'Solicita más descuento'
        else:
            viability = False
            message = 'La condición de compra es alta'
            utilities = 0
            porUtilities = 0

        print(message)
        queried = datetime.now(tz)
        result = {
            "viability": viability,
            "message": message,
            "suggest_sale_price": int(suggest_sale_price),
            "suggest_buy_price": int(suggest_buy_price),
            "saving": int(saving),
            "desc_price": int(desc_price),
            "invested_capital": int(inv_cap),
            "invest": int(invest),
            "total_saving": int(total_saving),
            "total_invest": int(total_inv),
            "profits": int(profits),
            "utilities": int(utilities),
            "percent_utilities": int(porUtilities),
        }

        query['desc'] = query['desc'] * 100
        time_id = time.time() * 100

        plt.figure(figsize=(10, 5))
        plt.plot(invests, 'b-', label='inverst')
        plt.plot([month1, month1], [invests[0], invests[len(invests)-1]],
                 'g--', label='profits starts')
        plt.plot([month2, month2], [invests[0],
                 invests[len(invests)-1]], 'r--', label='profits ends')
        plt.legend()
        posible_sales_URL = upload_img(
            cloud_path, f'/{time_id}-posibles_sale.jpg', plt)

        doc_ref.collection('providers_offers').document(query['provider']).set({
            "queried": queried,
            "result": result,
            "query": query,
            "posible_sales_URL": posible_sales_URL
        })

        return {
            "result": result,
            "status": 200,
            "message": "ok",
        }, 200


class Predictions():
    def by_year_sales(dataset):

        # dataset = formatValues(dataset)

        X = dataset[['Unitario Venta']]
        Y = dataset['Unidades']

        sc_X = StandardScaler()

        X = sc_X.fit_transform(X)
        X = sc_X.transform(X)

        # Entrenación
        reg = LinearRegression().fit(X, Y)
        print("The Linear regression score on training data is ",
              round(reg.score(X, Y), 2))

        # Basado en la cantidad de meses obtenidos, se ajusta para predecir al menos 1 año
        repeats = (12 - len(X)) / int(len(X))
        repeats = math.ceil(repeats) + 1
        X = np.tile(X, (repeats, 1))

        # crea lista de predicciones
        predict_year = reg.predict(X)
        year_df = pd.DataFrame(predict_year)
        year_df.fillna(value=0, inplace=True)
        df_file = year_df.to_csv(encoding='utf-8', index=False)

        # json.dump(p, codecs.open(local_path+'year_predictions.json', 'w'))
        # print(cloud_path)
        # print(df_file)
        yearpredictionsURL = upload_file(
            cloud_path+'/', 'year_predictions.csv', df_file)
        print(yearpredictionsURL)
        print('preicciones ok')

        return predict_year, reg, X, yearpredictionsURL

    def by_stats(dataset, test_size, window_size, product_name):
        # dataset = formatValues(dataset)
        dataset['Fecha'] = pd.to_datetime(dataset['Fecha'])
        dataset = dataset.set_index('Fecha')

        df_shift = dataset['Unidades'].shift(1)
        df_mean_roll = df_shift.rolling(window_size).mean()
        df_std_roll = df_shift.rolling(window_size).std()
        df_mean_roll.name = "mean_roll"
        df_std_roll.name = "std_roll"
        df_mean_roll.index = dataset.index
        df_std_roll.index = dataset.index

        df_w = pd.concat(
            [dataset['Unidades'], df_mean_roll, df_std_roll], axis=1)

        df_w = df_w[window_size:]

        test_cant = int(
            (dataset['Unidades'].describe()['count'])*(test_size*.01))

        test = df_w[-test_cant:]
        train = df_w[:-test_cant]
        X_test = test.drop("Unidades", axis=1)
        y_test = test["Unidades"]
        X_train = train.drop("Unidades", axis=1)
        y_train = train["Unidades"]

        try:
            clf = SVR(gamma="scale")
            clf.fit(X_train, y_train)
            y_train_hat = pd.Series(clf.predict(X_train), index=y_train.index)
            y_test_hat = pd.Series(clf.predict(X_test), index=y_test.index)
            hat_groups = y_test_hat.groupby(pd.Grouper(freq='M'))
        except:
            raise Exception(
                'No hay diferencia en los datos para realizar predicciones')

        total_predicted = np.sum(y_test_hat)
        mean_predicted = y_test_hat.describe()['mean']
        print(y_test_hat)
        chart_data = {
            "train": y_train,
            "prediction": y_test_hat,
            "test": y_test,
        }
        print(chart_data)
        predict_df = pd.DataFrame(data=chart_data)
        estimated_predict = predict_df.to_json(orient="columns")
        estimated_predict_JSON = upload_file(
            cloud_path, '/estimated_predict.json', estimated_predict)

        plt.figure(figsize=(12, 6))
        plt.plot(y_train, label='Datos de entrenamiento')
        plt.plot(y_test_hat, label='Predicción')
        plt.plot(y_test, label='Datos reales')
        plt.legend(loc='best')
        plt.title('Predicción de ventas ' + product_name)
        estimated_predict_URL = upload_img(
            cloud_path, '/estimated_predict.jpg', plt)

        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y_test, y_test_hat)
        print('Margen de error: {}'.format(mse))

        result = {
            "total_predicted": int(total_predicted),
            "months_predicted": len(hat_groups),
            "avg_for_sell": float("{:.2f}".format(mean_predicted)),
            "error_mean": float("{:.2f}".format(mse)),
            "imgURL": estimated_predict_URL,
            "jsonURL": estimated_predict_JSON
        }

        return result
