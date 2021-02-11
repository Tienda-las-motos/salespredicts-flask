from flask import Flask, jsonify, request, Response
from flask import render_template
from flask_cors import CORS, cross_origin
from src.file import Table



import pandas as pd
import json
import os
import io

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

    @app.route('/file', methods=['GET', 'POST'])
    def fileuploader():
        if request.method == 'POST':
            return Table.upload(request)
        else:
            return render_template('upload.html')
        
    @app.route('/table', methods=['GET'])
    def get_table(): return Table.get_table(request)
    
    

    # To get data body in json, use request.json
    # To get data body in form, use request.form
    
    if __name__ == '__main__':
        app.run(debug=True, port=4000)
    return app
        
app = index()
# if __name__ == '__main__':
#     index()