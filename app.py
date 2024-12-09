from flask import Flask, request, render_template
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Load the model
model = joblib.load('knn_pipeline_model.joblib')

# Class labels
class_dict = {
    "0": "Insufficient_Weight",
    "1": "Normal_Weight",
    "2": "Obesity_Type_I",
    "3": "Obesity_Type_II",
    "4": "Obesity_Type_III",
    "5": "Overweight_Level_I",
    "6": "Overweight_Level_II"
}

# Set up logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            logging.debug(f"Received form data: {request.form}")

            gender = request.form['gender']
            logging.debug(f"Gender: {gender}")
            age = float(request.form['age'])
            logging.debug(f"Age: {age}")
            height = float(request.form['height'])
            logging.debug(f"Height: {height}")
            weight = float(request.form['weight'])
            logging.debug(f"Weight: {weight}")
            family_history_with_overweight = request.form['family_overweight']
            logging.debug(f"Family History: {family_history_with_overweight}")
            favc = request.form['favc']
            logging.debug(f"FAVC: {favc}")
            fcvc = float(request.form['fcvc'])
            logging.debug(f"FCVC: {fcvc}")
            ncp = float(request.form['ncp'])
            logging.debug(f"NCP: {ncp}")
            caec = request.form['caec']
            logging.debug(f"CAEC: {caec}")
            smoke = request.form['smoke']
            logging.debug(f"Smoke: {smoke}")
            ch2o = float(request.form['ch2o'])
            logging.debug(f"CH2O: {ch2o}")
            scc = request.form['scc']
            logging.debug(f"SCC: {scc}")
            faf = float(request.form['faf'])
            logging.debug(f"FAF: {faf}")
            tue = float(request.form['tue'])
            logging.debug(f"TUE: {tue}")
            calc = request.form['calc']
            logging.debug(f"Calc: {calc}")
            mtrans = request.form['mtrans']
            logging.debug(f"MTRANS: {mtrans}")

            # Create DataFrame with appropriate data types
            data = {
                "gender": gender,
                "age": age,
                "height": height,
                "weight": weight,
                "family_history_with_overweight": family_history_with_overweight,
                "favc": favc,
                "fcvc": fcvc,
                "ncp": ncp,
                "caec": caec,
                "smoke": smoke,
                "ch2o": ch2o,
                "scc": scc,
                "faf": faf,
                "tue": tue,
                "calc": calc,
                "mtrans": mtrans
            }

            # Verify no missing values
            for key, value in data.items():
                if pd.isnull(value):
                    logging.error(f"Missing value for {key}: {value}")
                    return f"Error: Missing value for {key}", 400

            # Create DataFrame and ensure data types are correct
            my_df = pd.DataFrame([data])

            logging.debug(f"Input DataFrame: {my_df}")

            prediction = model.predict(my_df)[0]
            prediction_str = str(prediction)
            logging.debug(f"Prediction (as string): {prediction_str}")

            pred_class = class_dict[prediction_str]
            logging.debug(f"Prediction: {pred_class}")

            return render_template("index.html", prediction=pred_class)
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return str(e), 400
    else:
        return render_template("index.html", prediction=None)

if __name__ == '__main__':
    app.run(debug=True)