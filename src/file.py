# from IPython.core.display import display
from fbclient import FirebaseApp

import pandas as pd
import json
import os
import io

db = FirebaseApp.fs
st = FirebaseApp.st


class Table():
    def upload(request):

        # LOAD FILE
        # print(request.files['dataset'])

        try:
            uploaded_file = request.files['dataset']
        except:
            return {
                'message': 'La petición no contiene archivo'
            }, 400

        global synonyms_cols
        try:
            request.args['synonyms']
        except:
            print('no sinónimos')
            synonyms_cols = {}
        else:
            synonyms_cols = request.data['synonyms']

        try:
            print(synonyms_cols)
            filename = uploaded_file.filename
            namews = filename.replace(' ', '_')
            # current_directory = os.path.abspath(os.path.dirname(__file__))+'/'
            # local_path = 'api/uploads/'+filename
            print('Archivo ok')

            # READ FILE
            df = pd.read_csv(uploaded_file,  decimal=".",
                             header=0, thousands=r",")
            # df = validate_file_struct(df, synonyms_cols)

            # print(df.head())

            # number_cols = ['Unidades', 'Unitario Venta', 'Ventas', 'Unitario Costo', 'Costos', 'Margen Monto', 'Margen Porcentaje']
            # for column in number_cols:
            #     try:
            #         df[column] = df[column].dropna().astype(int).replace(',', '', regex=True)
            #     except ValueError:
            #         pass

            # FILTER THE LIST
            products_list = df[['Codigo', 'Descripcion']]
            products_list = products_list.drop_duplicates(
                subset=['Codigo']).dropna()
            products_list['Descripcion'] = products_list['Descripcion'].str.strip()

            # GENERATE JSON RESULT but don't upload in firebase storeage
            loadfile_result = products_list.to_json(orient="table")
            count = products_list.describe()['Codigo']['count']

            print('Lista creada')

            # STORAGE IN FIREBASE
            # Agregamos los datos válidos para obtener un id de firestore
            tables_ref = db.collection(u'tables')
            doc = tables_ref.add({
                'total_count': int(count),
                'file_name': filename,
            })
            doc_id = doc[1].id
            print('Id obtenido')

            # UPLOAD FILE CSV TO STORAGE
            cloud_path = 'tables/'+doc_id+'/'
            df.fillna(value=0, inplace=True)
            df_file = df.to_csv()
            fileURL = upload_file(cloud_path, namews, df_file)
            print('Archivo cargado a storage')

            result_data = {
                'total_count': int(count),
                'fileURL': fileURL,
                'file_name': filename,
                'doc_id': doc_id,
                'storage_path': cloud_path + namews
            }

            print('Documento actualizado')
            tables_ref.document(doc_id).update(result_data)

            result = {
                'data': result_data,
                'product_list': json.loads(loadfile_result)
            }

            return {
                'status': 201,
                'message': 'ok',
                'result': result
            }, 201

        except:
            return {
                'status': 400,
                'message': 'Error al leer el archivo',
            }, 400

    def get_table(request):
        # VALIDATE THERE IS TABLE ID
        try:
            table_id = request.args['id']
            doc_ref = db.collection(u'tables').document(table_id)
        except:
            return {
                'message': 'La petición debe incluir el parámetro table'
            }, 400

        # VALIDATE IS DOCUMENT IN FIRESTORE
        try:
            doc = doc_ref.get()
            result_data = doc.to_dict()
            doc_URL = doc.to_dict()['fileURL']
        except:
            return {
                'message': 'El archivo no exite o fue eliminado',
                'status': 400
            }, 400

        # FILTER THE LIST
        df = pd.read_csv(doc_URL,  decimal=".")
        products_list = df[['Codigo', 'Descripcion']]
        products_list = products_list.drop_duplicates(
            subset=['Codigo']).dropna()
        products_list['Descripcion'] = products_list['Descripcion'].str.strip()

        # GENERATE JSON RESULT
        loadfile_result = products_list.to_json(orient="table")
        count = products_list.describe()['Codigo']['count']

        result = {
            'data': result_data,
            'product_list': json.loads(loadfile_result),
        }

        return {
            'message': 'ok',
            'result': result,
            'status': 200
        }, 200


def validate_file_struct(df, synonyms_cols):
    try:
        df.drop(columns='Unnamed: 0')
    except:
        print("No hay columnas sin nombre")
    global code_des
    global cols_must_be
    global needed_cols

    cols_must_be = ['Fecha', 'Unidades', 'Unitario Venta', 'Ventas', 'Unitario Costo',
                    'Costos', 'Margen Monto', 'Margen Porcentaje', 'Codigo', 'Descripcion']
    needed_cols = ['Fecha', 'Unidades', 'Unitario Venta', 'Ventas',
                   'Unitario Costo', 'Costos', 'Margen Monto', 'Margen Porcentaje']

    # revisar si las columnas vienen bien
    try:
        df[cols_must_be]
    except:
        # Las columnas no vienen bien, se buscará si hay sinónimos
        if len(synonyms_cols) > 0:
            for key, value in synonyms_cols.items():
                # Se intenta cambiar las columnas, si no es posible, se rechaza la petición
                try:
                    df = df.rename(columns={key: value})
                except:
                    return {
                        'message': f'fallo al intentar cambiar columna {key} a {value}',
                        'status': 404
                    }, 404

        # intenta revisar de nuevo las columnas
        try:
            df[cols_must_be]
        # si siguen sin estar bien, se revisa que la columna de codigo y descripción vengan fusionadas
        except:
            # primero revisamos que no haya error con las demás columnas
            try:
                df[needed_cols]
            except:
                # si hay error en las demás columnas revisamos en cuál y lo notificamos
                for col in needed_cols:
                    try:
                        df[col]
                    except:
                        return {
                            'message': f'no se encontró la columna {col}',
                            'status': 404
                        }, 404
            # si las columnas están bien, revisamos "CódigoInventario"
            else:
                try:
                    df['CódigoInventario']
                except:
                    return {
                        'message': 'archivo tampoco contiene columna "CódigoInventario"',
                        'status': 404
                    }, 404
                # si la columna estaba fusionada, la separamos
                else:
                    code_des = df['CódigoInventario'].str.split(
                        ' ', n=1, expand=True)
                    df['Codigo'] = code_des[0]
                    df['Descripcion'] = code_des[1]
                    df.drop(columns=['CódigoInventario'], inplace=True)
    else:
        print('Archivo contiene columnas correctas')

    return df


def upload_file(cloud_path, filename, file):
    # print(cloud_path+filename)
    bucket = st.bucket()
    cloud_file = bucket.blob(cloud_path+filename)
    cloud_file.upload_from_string(
        file, content_type='application/octet-stream')
    cloud_file.make_public()
    # default_storage.delete(local_path+filename)
    return cloud_file.public_url


def upload_img(cloud_path, filename, file):
    img_data = io.BytesIO()
    file.savefig(img_data, format="jpg")
    img_data.seek(0)

    bucket = st.bucket()
    cloud_file = bucket.blob(cloud_path+filename)
    cloud_file.upload_from_file(img_data, content_type='image/jpg')
    cloud_file.make_public()
    return cloud_file.public_url
