from flask import Flask, render_template, request
import pickle
import numpy as np
import os


app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


model_path = os.path.join(BASE_DIR, 'ridge.pkl')
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')

ridge_model = pickle.load(open(model_path, 'rb'))
standard_scaler = pickle.load(open(scaler_path, 'rb'))



@app.route("/")
def home():
    return render_template('home.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        try:
            # Get input values
            Temperature = float(request.form.get("Temperature"))
            RH = float(request.form.get("RH"))
            Ws = float(request.form.get("Ws"))
            Rain = float(request.form.get("Rain"))
            FFMC = float(request.form.get("FFMC"))
            DMC = float(request.form.get("DMC"))
            ISI = float(request.form.get("ISI"))
            Classes = float(request.form.get("Classes"))
            Region = float(request.form.get("Region"))

            
            input_data = np.array([[
                Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region
            ]])

          
            scaled_data = standard_scaler.transform(input_data)

           
            prediction = ridge_model.predict(scaled_data)[0]

            return render_template('home.html', result=round(prediction, 2))

        except Exception as e:
            return render_template('home.html', result=f"Error: {str(e)}")

    return render_template('home.html')



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)