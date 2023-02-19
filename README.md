# Sales Predictor (Tienda Las motos)
Aplicación backend de API's desarrollada en **Flask** para el análisis y predicción de ventas. Montada en el servicio Cloud Run de Google Cloud Platform.

**Versión:** 1.2

## REST
**Base URL:** https://sales-analysis-mbdlq5hwgq-wl.a.run.app
La base URL contiene la notificación de que el servicio está activo y la versión del mismo.

``` hs
GET https://sales-analysis-mbdlq5hwgq-wl.a.run.app
```
```py
RESPONSE 'Sales predictions API works on Version 1.2!'
```

### File Section
**Base URL:** https://sales-analysis-mbdlq5hwgq-wl.a.run.app/file

Endpoint para recibir un archivo CSV y agregarlo a la base de datos de **Firebase Firestore**, extrae como resultado la cantidad de productos y las url del archivo en el **Firebase Storage**

``` hs
GET https://sales-analysis-mbdlq5hwgq-wl.a.run.app/file
```
```json

```

## Developer Mode
Para realizar modificaciones en el proyecto deben cumplirse algunos requerimientos antes de correr la aplicación en local.

### Requerimientos
Se requiere una computadora con:
  - Git
  - Python 3

### Instrucciones
1. Para descargar el proyecto. Desde un bash o una consola de windows, realizar el comando ``git clone https://github.com/jgu7man/salespredicts-flask`
2. Se recomienda instalar [Mini Conda](https://docs.conda.io/en/latest/miniconda.html) o [ANACONDA](https://docs.anaconda.com/anaconda/install/) para establecer un *environment*.

3. Crear el environment en la raíz del proyecto de

    ```hs
    conda env create -f ./environment.yml
    ```

4. Activate el entorno

    ```hs
    conda activate sales_predictor
    ```


5. (OPCIONAL) Instalar las dependencias requeridas del documento  `requirements.txt`

    ```hs
    pip install -r requirements.txt
    ```

6. Ajustar las variables de flask

    **BASH**
    ```hs
    export FLASK_APP=main
    export FLASK_ENV=development
    ```

    **CMD**
    ```hs
    set FLASK_APP=hello
    set FLASK_ENV=development
    ```

7. Correr la aplicación
    ```
    flask run
     Running on http://127.0.0.1:5000/
    ```

## Testing
Para testeo de la api en versión local o cloud, puedes usar [postman](https://www.postman.com/downloads/)

Importa los archivos incluidos en el [folder test](./test). El folder incluye:
- La colección de API's
- El environment local
- El environment del server en GCP

## Deploy
```hs
gcloud run deploy --source .
```

## Dependencias
Esta aplicación requiere:
- Flask v2.1
- Firebase for python 4.4.0
