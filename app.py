#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Housing price prediction - estimates house value based on input values

Copyright (C) 2018 Risto Nykanen <risto.nykanen@iki.fi>

"""

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import json
import pickle
import pandas as pd
from flask import Flask, request, url_for, jsonify, render_template
from flask_json import FlaskJSON, JsonError, json_response, as_json
from calc_dev import calc_dev

app = Flask(__name__)
app.config['JSON_ADD_STATUS'] = False

FlaskJSON(app)

sample = {
  "crime_rate": [0.1],
  "avg_number_of_rooms": [4.0],
  "distance_to_employment_centers": [6.5],
  "property_tax_rate": [330.0],
  "pupil_teacher_ratio": [19.5]
}


#    "The service calculates housing price prediction based on area data."

@app.route('/')
def home():
    if (request.headers["User-Agent"].count("curl")):
        return "Usage, e.g: curl http://<server name>/predict -d <JSON-data>"
    else:
      return render_template('home.html')

@app.route('/help')
def api_help():
    if (request.headers["User-Agent"].count("curl")):
      return "Usage, e.g: curl http://<server name>/predict -d <JSON-data>"
    else:
      return render_template('home.html')

@app.route('/predict', methods = ['GET', 'POST'])
def api_predict():
    curl_agent = request.headers["User-Agent"].count("curl")
    
    # Give an advice
    if request.method == 'GET' and curl_agent:
        return("Usage, e.g: curl http://<server name>/predict -d " + str(sample))

    features = ['crime_rate', 'avg_number_of_rooms','distance_to_employment_centers', 'property_tax_rate','pupil_teacher_ratio']
    # Initialize data structure for input values 
    pre_values = pd.DataFrame(sample)
    pre_values = pre_values.reindex(columns=features)

    if (not curl_agent):
        result=request.form
        for value in features:
            pre_values[value] = float(result[value])
        print(pre_values)    
    else :    
    # Use 'force' to skip mimetype checking to have shorter curl command.
      try:
          data = request.get_json(force=True)
          for value in features:
              pre_values[value] = float(data[value])

      except (KeyError, TypeError, ValueError):
          raise JsonError(description='Invalid value.')

    # Load generated model
    pkl_file = open('reg_model.pkl', 'rb')
    regressor = pickle.load(pkl_file)
    result = (float) (regressor.predict(pre_values))

    # Load y-values for Std Dev calculation
    pkl_file = open('y_values.pkl', 'rb')
    y = pickle.load(pkl_file)

    # Standard deviation of estimated value, using x nearest data points for inputs
    stddev = calc_dev(pre_values, 20)
    
    if curl_agent : 
      return json_response(house_value = result, stddev = stddev)
    else :
      return render_template('result.html',house_value = result, stddev = stddev)

       
if __name__ == '__main__':
    app.debug = True
    app.run()

