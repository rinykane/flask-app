import json
import pickle
import pandas as pd
from flask import Flask, request, url_for, jsonify, render_template
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
},{
  "crime_rate": 0.2,
  "avg_number_of_rooms": 5.0,
  "distance_to_employment_centers": 7.5,
  "property_tax_rate": 300.0,
  "pupil_teacher_ratio": 31.5
}


#@app.route('/')
#def api_root():
#    return "The service calculates housing price prediction based on area data."

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

    Housing=pd.read_csv('housing.csv')
    features = ['crime_rate', 'avg_number_of_rooms','distance_to_employment_centers', 'property_tax_rate','pupil_teacher_ratio']

    y = Housing['house_value']
    X = Housing[features]

    pre_values = pd.DataFrame(X.iloc[0:1])
#    pre_values = pd.DataFrame(sample[0:1])

    if (not curl_agent):
        result=request.form
        for value in features:
            pre_values[value] = float(result[value])
        print(pre_values)    
    else :    
    # Use 'force' to skip mimetype checking to have shorter curl command.
#    print("Getting json input...")
      try:
          data = request.get_json(force=True)
          # Use same DF shape
#          pre_values = pd.DataFrame(sample)
#          print(pre_values)
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
    
    if curl_agent : 
      return json_response(house_value = result, stddev = stddev)
    else :
      return render_template('result.html',house_value = result, stddev = stddev)


       
if __name__ == '__main__':
    app.debug = True
    app.run()

