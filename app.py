import json
import pickle
import pandas as pd
from flask import Flask, request, url_for, jsonify
from flask_json import FlaskJSON, JsonError, json_response, as_json

app = Flask(__name__)
app.config['JSON_ADD_STATUS'] = False

FlaskJSON(app)

sample = {
  "crime_rate": 0.1,
  "avg_number_of_rooms": 4.0,
  "distance_to_employment_centers": 6.5,
  "property_tax_rate": 330.0,
  "pupil_teacher_ratio": 19.5
}


@app.route('/')
def api_root():
    return 'The service calculates housing price prediction based on area data.'

@app.route('/help')
def api_help():
    return 'Usage, e.g: curl http://<URL>/predict -d <JSON-data>'

@app.route('/predict', methods = ['GET', 'POST'])
def api_predict():
    # Give an advice
    if request.method == 'GET':
        return("Usage, e.g: curl http://<URL>/predict -d " + str(sample))

    Housing=pd.read_csv('housing.csv')
    features = ['crime_rate', 'avg_number_of_rooms','distance_to_employment_centers', 'property_tax_rate','pupil_teacher_ratio']

    y = Housing['house_value']
    X = Housing[features]
        
    # Use 'force' to skip mimetype checking to have shorter curl command.
#    print("Getting json input...")
    try:
        data = request.get_json(force=True)
#        print(data)    
        # Use same DF shape
        pre_values = pd.DataFrame(X.iloc[0:1])
#        pre_values = pd.DataFrame(sample)
#        print(pre_values)
        for value in features:
            pre_values[value] = float(data[value])

    except (KeyError, TypeError, ValueError):
        raise JsonError(description='Invalid value.')

#    print("Loading model...")
    # Load generated model
    pkl_file = open('reg_model.pkl', 'rb')
    regressor = pickle.load(pkl_file)
    result = (float) (regressor.predict(pre_values))

#    print("Prediction: " + str(result))
    # Standard deviation of x nearest data points
#        pkl_file = open('y_values', 'rb')
#        y = pickle.load(pkl_file)
#    print(y)
    find_closest = y.loc[(y-result).abs().argsort()[:20]]
#    print(find_closest)
    stddev = find_closest.std()
#    print("Stddev: " + str(stddev))
             
    return json_response(house_value = result, stddev = stddev)

       
if __name__ == '__main__':
    app.debug = True
    app.run()

