from flask import Flask, jsonify, request, Response
from flask import render_template
from flask_cors import CORS, cross_origin
from src.file import Table
from src.products import Product
from src.predictions import SalesPredictions, Analyze


def index():
    
    
    
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a Response object.
    """
    app = Flask(__name__)
    CORS(app)
    cors = CORS(app, resources={
        r"/":{
            "origins":"*"
        }
    })
    
    @app.route('/', methods=['GET', 'POST'])
    def test():
        return 'Sales predictions API works on Version 1.2!'

    @app.route('/file', methods=['GET', 'POST'])
    def fileuploader():
        if request.method == 'POST':
            return Table.upload(request)
        else:
            return render_template('upload.html')
        
    @app.route('/table', methods=['GET'])
    def get_table(): return Table.get_table(request)
    
    
    @app.route('/product/filter', methods=['GET'])
    def product(): return Product.filter(request)


    @app.route('/predictions/<string:query>', methods=['GET', 'POST'])
    def predictions(query):
        if query == 'sales-months':
            return SalesPredictions.cant_query(request)
        elif query == 'sales-cant':
            return SalesPredictions.months_query(request)
        elif query == 'sales-stats':
            return SalesPredictions.stats_query(request)
        elif query == 'sales-seasonal':
            return SalesPredictions.seasonal(request)
        else: 
            return {
                'message': 'Error en la ruta', 
                'status': 404
            }, 404
    
    
    @app.route('/analyze/<string:criteria>', methods=['GET', 'POST'])
    def analyze(criteria):
        print(criteria)
        if criteria == 'provider-offering':
            return Analyze.providers_offers(request)
        else: 
            return {
                'message': 'Error en la ruta', 
                'status': 404
            }, 404


    # To get data body in json, use request.json
    # To get data body in form, use request.form
    
    if __name__ == '__main__':
        app.run(debug=False, port=5000)
    return app
        
app = index()
# if __name__ == '__main__':
#     index()