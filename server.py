from flask import Flask, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run-model/<rRatio>/<expLvl>/<compSize>/<compLoc>/<empType>/<jobTitle>')
def run_model(rRatio='',expLvl='',compSize='',compLoc='',empType='',jobTitle=''):
    features = ['experience_level', 'remote_ratio', 'company_size', 'job_title', 'company_location', 'employment_type']
    all_features = ['experience_level', 'remote_ratio', 'company_size', 'job_title', 'company_location',
                    'employment_type', 'salary_in_usd']
    s_features = ['experience_level', 'company_size', 'job_title', 'company_location', 'employment_type']

    originalDataframe = pd.read_csv("ds_salaries.csv")[all_features]
    salaryQ1 = originalDataframe['salary_in_usd'].quantile(0.25)
    salaryQ3 = originalDataframe['salary_in_usd'].quantile(0.75)
    salaryOutliers = originalDataframe[(originalDataframe['salary_in_usd'] > salaryQ3 + 1.5 * (salaryQ3 - salaryQ1))]
    originalDataframe = originalDataframe.drop(salaryOutliers.index)
    outputDataframe = originalDataframe['salary_in_usd']
    outputScaler = StandardScaler()
    outputScaler.fit(outputDataframe.to_numpy().reshape(-1, 1))

    theInput = pd.DataFrame([[expLvl, int(rRatio), compSize, jobTitle, compLoc, empType]], columns=features)

    print(theInput)

    inputDataframe = originalDataframe[features]
    inputDataframe = pd.concat([inputDataframe, theInput], ignore_index=True)

    inputDataframe = pd.get_dummies(inputDataframe, columns=s_features)

    originalColumns = pd.get_dummies(originalDataframe[features], columns=s_features)

    if (len(inputDataframe.columns) != len(originalColumns.columns)):
        print("Mistmatch in columns!")
        print(inputDataframe.columns)
        print(originalColumns.columns)
        for col in inputDataframe.columns:
            if col not in originalColumns.columns:
                print("New column! [",str(col),"]")
        return("0")

    scaler = StandardScaler()
    scaler.fit(inputDataframe.to_numpy())
    inputDataframe = scaler.transform(inputDataframe.to_numpy())

    best_model = pickle.load(open("best_model.sav", 'rb'))

    prediction = 0
    try:
        prediction = outputScaler.inverse_transform([best_model.predict([inputDataframe[-1]])])[0][0]
    except:
        return("Sorry, this model has not seen that job title before. Please try a different one!")
    return ('Expected Salary:'+str(prediction))

if __name__ == '__main__':
    app.run(debug=True)