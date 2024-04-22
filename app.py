from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pickled model
with open('artifacts/sgd_best_model.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        gender = int(request.form['gender'])
        age = int(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = int(request.form['ever_married'])
        work_type = int(request.form['work_type'])
        Residence_type = int(request.form['residence_type'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = int(request.form['smoking_status'])

        # Create DataFrame
        data = {
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'ever_married': [ever_married],
            'work_type': [work_type],
            'Residence_type': [Residence_type],
            'avg_glucose_level': [avg_glucose_level],
            'bmi': [bmi],
            'smoking_status': [smoking_status]
        }
        imp_df = pd.DataFrame(data)
        print(imp_df)

        # Make predictions
        prediction = model.predict(imp_df)

        # Pass prediction to the template
        return render_template('output.html', prediction=prediction[0])

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
