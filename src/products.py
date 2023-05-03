# from IPython.core.display import display
import json
import locale
import os
import warnings

import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from fbclient import FirebaseApp
from src.file import upload_file, upload_img

# warnings.filterwarnings("ignore")


db = FirebaseApp.fs
st = FirebaseApp.st
mlp.style.use('seaborn')


class Product():
    def filter(request):

        # Validate there is table id param 
        print(request.args['table'])
        global table_id
        try:
            table_id = request.args['table']
            doc_ref = db.collection(u'imported_tables').document(table_id)
        except:
            return {
                'message': 'La petición debe incluir le parámetro "table"',
                'status': 400
            }, 400

        # Validate is product id param
        try:
            product_id = request.args['product']
        except:
            return {
                'message': 'La petición debe incluir el parámetro "product"',
                'status': 400
            }, 400

        # Validate if is the document in firestore
        try:
            doc = doc_ref.get()
            doc_URL = doc.to_dict()['fileURL']
        except:
            return {
                'message': 'El archivo no exite o fue eliminado',
                'status': 404
            }, 404

        # read the document
        df = pd.read_csv(doc_URL,  decimal=".", thousands=",")
        print(df.head())
        df = formatValues(df)

        # Validate if is the product in list
        try:
            product_selected = df.loc[df['Codigo'] == product_id]
        except:
            return {
                'message': 'Error al intentar elegir el producto en esta tabla',
                'status': 400
            }, 400

        # locale.setlocale(locale.LC_TIME, 'es_MX.UTF-8')

        print('petición realizada con éxito')
        print(product_selected.head())

        # Define paths
        global product_name
        global product_path
        global local_path
        global current_directory
        product_name = product_selected['Descripcion'].unique()[0].strip()
        product_refname = product_name.replace(' ', '_').lower()
        product_path = 'tables/'+table_id+'/products/'+product_id+'/'
        current_directory = os.path.abspath(os.path.dirname(__file__))+'/'
        local_path = 'api/uploads/'

        # Get stats
        product_stats = get_product_stats(product_selected)
        print('product_stats ok')
        sell_stats = get_sell_stats(product_selected)
        print('sell_stats ok')
        time_stats = get_sales_timeline(product_selected)
        print('time_stats ok')
        buy_stats = get_buy_stats(product_selected)
        print('buy_stats ok')

        # Storage files
        ps_file = product_selected.to_csv()
        datasetURL = upload_file(product_path, 'product_dataset.csv', ps_file)
        print('dataset uploaded')
        product_ref = doc_ref.collection('products').document(product_id)
        product_ref.set({
            "name": product_name,
            "code": product_id,
            "dataset": datasetURL,
            "product_stats": product_stats,
            "buy_stats": buy_stats,
            "sell_stats": sell_stats,
            "time_stats": time_stats
        })
        print('firestore updated')

        return {
            'status': 200,
            'message': 'ok',
            'result': {
                "name": product_name,
                "code": product_id,
                "dataset": datasetURL,
                "product_stats": product_stats,
                "buy_stats": buy_stats,
                "sell_stats": sell_stats,
                "time_stats": time_stats
            }
        }, 200


def get_product_stats(dataset):

    print(dataset.head())
    print(dataset['Unitario Costo'].head())
    print(dataset['Unitario Costo'].describe(include='all'))
    
    # Calculate promediates
    avg_margin = dataset['Margen Porcentaje'].describe()['mean']
    avg_buy_price = dataset['Unitario Costo'].describe()['mean']
    avg_sale_price = dataset['Unitario Venta'].describe()['mean']
    max_margin = dataset['Margen Porcentaje'].describe()['max']
    max_sale_price = dataset['Unitario Venta'].describe()['max']
    min_buy_price = dataset['Unitario Costo'].describe()['min']
    sales_quantity = dataset['Unidades'].describe()['count']
    print('stats created')

    sold_units = dataset['Unidades'].sum()
    sold_units = int(sold_units)

    stats = {
        "sold_units": int(sold_units),
        "sales_quantity": int(sales_quantity),
        "avg_sale_price": int(avg_sale_price),
        "max_sale_price": int(max_sale_price),
        "avg_margin": int(avg_margin * 100),
        "max_margin": int(max_margin * 100),
        "avg_buy_price": int(avg_buy_price),
        "min_buy_price": int(min_buy_price),
    }
    return stats


def get_sell_stats(dataset):
    # Filtra las columnas necesarias
    precio_venta_list = dataset.groupby(
        dataset['Unitario Venta'],
        as_index=False).aggregate({
            'Unidades': 'sum',
            'Margen Porcentaje': 'mean'
        })

    # Precio de venta con mayor rendimiento
    max_venta_margen = precio_venta_list['Margen Porcentaje'].describe()['max']
    max_venta_precio_row = precio_venta_list[precio_venta_list['Margen Porcentaje']
                                             == max_venta_margen]
    max_venta_precio_margen = max_venta_precio_row['Unitario Venta'].values[0]

    # Rendimiento promedio
    avg_margen = precio_venta_list['Margen Porcentaje'].describe()['mean']

    print('sell stats getted')

    #SECTION - Obtener precio sugerido de venta mediante regresión lineal

    X = precio_venta_list[['Margen Porcentaje']]
    Y = precio_venta_list['Unitario Venta']
    X = precio_venta_list[['Margen Porcentaje']]
    # saleY_test = precio_venta_list['Unitario Venta']

    #TODO - Search what do this
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X.values)
    X = sc_X.transform(X)

    # Entrenamiento de datos
    reg = LinearRegression().fit(X, Y)
    score_error2 = reg.score(X, Y)
    print('sell stats trained')

    # Predicción   
    predict = reg.predict(X)
    suggest_sale_price = reg.predict([[avg_margen]])
    print('sell predictions ok')

    # Obtiene imagen del gráfico
    plt.figure()
    plt.plot(predict, 'ro', suggest_sale_price, 'bo')
    suggessalepriceURL = upload_img(
        product_path, 'suggest_sale_price.jpg', plt)

    # !SECTION

    return {
        'max_throwput_sale_price': int(max_venta_precio_margen), # precio de venta de  mayor rendimiento
        'avg_throwput_sale': int(avg_margen), # venta de rendimiento promedio
        # 'score_error2': int(score_error2), # margen de error
        'suggest_sale_price': int(suggest_sale_price), # precio sugerido de venta
        "suggest_sale_price_img": suggessalepriceURL, # gráfico de precio sugerido de venta
        "avg_sale_price": precio_venta_list.describe()['Unitario Venta']['mean'], # precio de venta promedio
        "max_sale_price": precio_venta_list.describe()['Unitario Venta']['max'] # precio de venta máximo 
    }


def get_buy_stats(dataset):
    # Filtra las columnas necesarias
    precio_compra_list = dataset.groupby(
        dataset['Unitario Costo'],
        as_index=False).aggregate({
            'Unidades': 'sum',
            'Margen Porcentaje': 'mean',
            'Unitario Venta': 'mean'
        })

    # avg_buy_price = precio_compra_list['Unitario Costo'].describe()['mean']
    avg_buy_margen = precio_compra_list['Margen Porcentaje'].describe()['mean']
    print('buy stats getted')

    #SECTION - Precio sugerido de venta mediante regresión lineal
    X = precio_compra_list[['Margen Porcentaje']]
    Y = precio_compra_list['Unitario Costo']

    sc_X = StandardScaler()

    X = sc_X.fit_transform(X.values)
    X = sc_X.transform(X)

    # Entrenamiento de datos
    reg = LinearRegression().fit(X, Y)
    error_score2 = reg.score(X, Y)
    print('buy stats trained')

    # Predicción
    predict = reg.predict(X)
    suggest_buy_price = reg.predict([[avg_buy_margen]])
    suggest_buy_price = suggest_buy_price[0]
    print('buy stats predicted')

    # Gráfico de la regression lineal
    plt.figure()
    plt.plot(predict, 'ro', suggest_buy_price, 'bo')
    suggestbutpriceURL = upload_img(product_path, 'suggest_buy_price.jpg', plt)
    print('but stats chart created')
    #!SECTION

    return {
        # "error_score2":error_score2, # margen de error
        "suggest_buy_price": int(suggest_buy_price), # precio sugerido de venta mediante regresión lineal
        "suggest_buy_price_URL": suggestbutpriceURL,  # gráfico del precio sugerido de venta mediante regresión
        'avg_buy_price': precio_compra_list.describe()['Unitario Venta']['mean'], # precio promedia de compra
        "max_buy_price": precio_compra_list.describe()['Unitario Venta']['min'] # precio menor de compra
    }


def get_sales_timeline(dataset):

    # Get required columns
    sales_timeline = dataset.groupby(dataset['Fecha'], as_index=True).aggregate({
        'Unidades': 'sum',
        'Ventas': 'sum',
        'Costos': 'sum',
    })

    """ First row of sales timeline"""
    first = sales_timeline.iloc[0].name
    """ Last row of sales timeline"""
    last = sales_timeline.iloc[-1].name

    # Tranform the timeline data and storages in a CSV file
    timeline_df = sales_timeline.to_csv()
    timelineURL = upload_file(product_path, 'sales-timeline.csv', timeline_df)
    print('timeline json created')

    # SECTION - Get stats for each month
    timestats = get_timestats(dataset)

    unitsbymonth = get_salesvscosts(sales_timeline)

    monthsbox = get_boxmonths(sales_timeline)
    #!SECTION

    # BUILD RESULT
    result = {
        "max_monthsales": int(monthsbox['maxlength']),
        "avgsales_per_month": float("{:.2f}".format(monthsbox['avg_mes'])),
        "first_sale": first,
        "last_sale": last,
        "max_sales_month": timestats['sales_months'],
        "max_throwput_month": timestats['margen_months'],
        "files": {
            "salesvscosts_chart_URL": unitsbymonth['salesvscosts_chart_URL'],
            "unitsbymonths_df_URL": unitsbymonth['unitsbymonths_df_URL'],
            "timeline": timelineURL,
            "meses_list_df": timestats['meses_list_df'],
            "boxchart_URL": monthsbox['boxchart_URL'],
        },
    }

    return result


def get_salesvscosts(dataset):
    # Group data by sales in date
    df_dates = dataset.groupby(
        dataset.index,
        as_index=True).aggregate({
            'Unidades': 'sum'
        })

    # Agrupa las ventas por semanas
    df_periods = dataset.groupby(pd.Grouper(freq='W')).aggregate({
        # 'Unidades': 'sum',
        'Ventas': 'sum',
        'Costos': 'sum',
    })

    # Crea el gráfico de ventas vs costos
    plt.figure()
    df_periods.plot(figsize=(14, 6),  title='Ventas vs Costos', )
    # df_periods['Costos'].plot(figsize = (14,6), lw=2, label="Costos")

    # df_dates.plot( figsize=(12, 5), title="Unidades vendidas por mes");
    salesvscosts_chart_URL = upload_img(product_path, 'salesvscosts.jpg', plt)
    print('sales normalized jpg created')

    dates_df = df_dates.to_csv(header=False)
    unitsbymonths_df_URL = upload_file(
        product_path, 'unitsbymonths.csv', dates_df)
    print('unitsbymonths csv created')

    return {
        "salesvscosts_chart_URL": salesvscosts_chart_URL,
        "unitsbymonths_df_URL": unitsbymonths_df_URL,
    }


def get_boxmonths(dataset):

    # Group data by month
    product_dataset = dataset['Unidades']
    months = product_dataset.groupby(pd.Grouper(freq='M'))
    print('months sales grouped')


    indexes = []
    datas = []
    lengths = []
    sales = []
    for name, group in months:
        groupSales = []
        for value in group.values:
            if value != 0:
                groupSales.append(value)
        sales.append(len(groupSales))
        datas.append(group.values)
        lengths.append(len(group.values))
        indexes.append(name.strftime('%B'))

    maxlength = max(lengths)
    transactions = np.sum(sales)
    avg_mes = transactions/len(months)
    print('time stats getted')

    # NORMALIZE DATAFRAME
    normalized_df = pd.DataFrame()
    for mes, (name, group) in zip(indexes, months):
        values = group.to_list()
        less = maxlength - len(group.values)

        for zero in range(less):
            values.append(zero * 0)

        normalized_df[mes] = pd.Series(values)

    print('normalized dataframe created')
    
    # MAKE CHARTS IMAGES
    plt.figure()
    months_box = normalized_df.replace(0, np.nan)
    boxes = months_box.boxplot(figsize=(8, 5))

    # boxes.figure.savefig(current_directory+"boxes-chart.jpg", format="jpg")
    boxchart_URL = upload_img(product_path, 'boxes-chart.jpg', boxes.figure)
    print('sales chart created')

    return {
        "maxlength": maxlength,
        "avg_mes": avg_mes,
        "boxchart_URL": boxchart_URL
    }


def get_timestats(dataset):
    # Agupa los datos por mes
    ps_dates = dataset.set_index('Fecha')
    meses_list = ps_dates.groupby(pd.Grouper(freq="M")).aggregate({
        'Unidades': 'sum',
        'Unitario Venta': 'mean',
        'Unitario Costo': 'mean',
        'Ventas': 'sum',
        'Costos': 'sum',
        'Margen Monto': 'sum',
        'Margen Porcentaje': 'mean',
    }).dropna()
    print('months list grouped')

    # Obtiene lo meses con más ventas
    max_sales = meses_list['Unidades'].describe()['max']
    max_sales_month = meses_list[meses_list['Unidades'] == max_sales]
    sales_months = []
    for month, row in max_sales_month.iterrows():
        str_month = month.strftime('%B')
        sales_months.append(str_month)
    print('max sale months getted')

    # Obtiene el mes con mayor rendimiento
    max_throwput_month = meses_list['Margen Porcentaje'].describe()['max']
    max_margen_month = meses_list[meses_list['Margen Porcentaje']
                                  == max_throwput_month]
    margen_months = []
    for month, row in max_margen_month.iterrows():
        str_month = month.strftime('%B')
        margen_months.append(str_month)
    print('max throwput months getted')

    monthlist_df = meses_list.to_csv()
    meseslistURL = upload_file(product_path, 'month-sales.csv', monthlist_df)
    print('month sales csv created')

    return {
        "sales_months": sales_months,
        "margen_months": margen_months,
        "meses_list_df": meseslistURL
    }


def formatValues(dataset):
    if 'Fecha' in dataset.columns:
        dataset['Fecha'] = pd.to_datetime(
            dataset['Fecha'], format='%d/%m/%Y', errors='coerce')
    if 'Unidades' in dataset.columns:
        dataset['Unidades'] = pd.to_numeric(
            dataset['Unidades'], errors='coerce')
    if 'Unitario Venta' in dataset.columns:
        dataset['Unitario Venta'] = pd.to_numeric(
            dataset['Unitario Venta'], errors='coerce')
    if 'Ventas' in dataset.columns:
        dataset['Ventas'] = pd.to_numeric(dataset['Ventas'], errors='coerce')
    if 'Unitario Costo' in dataset.columns:
        dataset['Unitario Costo'] = pd.to_numeric(
            dataset['Unitario Costo'], errors='coerce')
    if 'Costos' in dataset.columns:
        dataset['Costos'] = pd.to_numeric(dataset['Costos'], errors='coerce')
    if 'Margen Monto' in dataset.columns:
        dataset['Margen Monto'] = pd.to_numeric(
            dataset['Margen Monto'], errors='coerce')
    if 'Margen Porcentje' in dataset.columns:
        dataset['Margen Porcentje'] = pd.to_numeric(
            dataset['Margen Porcentje'], errors='coerce')

    return dataset
