import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.preprocessing import OneHotEncoder
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle
from datetime import datetime
import os

app = Flask(__name__, static_folder='static')

dataset=None
model=None
x_train=None
y_train=None
x_test=None
y_test=None
current_model_file = None
attributes = []
encoder = None
saved_models=[]

def calculate_accuracy(algorithm, x_train, x_test, y_train, y_test):
    if algorithm == 'random_forest':
        model= RandomForestClassifier()
    elif algorithm == 'knn':
        model = KNeighborsClassifier()
    elif algorithm == 'svc':
        model = SVC()
    else:
        return "Invalid Algorithm Selected"
    
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    return accuracy

def calculate_metrics(algorithm, x_train, x_test, y_train, y_test):
    if algorithm == 'random_forest':
        model = RandomForestClassifier()
    elif algorithm == 'knn':
        model = KNeighborsClassifier()
    elif algorithm == 'svc':
        model = SVC()
    else:
        return "Invalid Algorithm Selected"

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = model.score(x_test, y_test)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    return accuracy, precision, recall, f1

def get_model(algorithm):
    if algorithm == 'random_forest':
        return RandomForestClassifier()
    elif algorithm == 'knn':
        return KNeighborsClassifier()
    elif algorithm == 'svc':
        return SVC()
    else:
        raise ValueError('Invalid algorithm selected')
    
def save_model(model, filename):
    global current_model_file
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    current_model_file = filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load', methods=['GET', 'POST'])
def load_dataset():
    global dataset, encoder
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('load.html', message='No File Uploaded!')

        file = request.files['file']
        if file.filename == '':
            return render_template('load.html', message='No File Uploaded!')

        dataset = pd.read_csv(file)

        # Handling missing values
        dataset.fillna(dataset.mean(), inplace=True)

        x = dataset.iloc[:, :-1]  # Use all columns except the last one as features
        y = dataset.iloc[:, -1]   # Use the last column as the target variable

        return render_template('load.html', message='Dataset loaded successfully!')

    return render_template('load.html')


@app.route('/split', methods=['GET', 'POST'])
def split_dataset():
    global dataset, x_train, x_test, y_train, y_test
    x = dataset.iloc[:, :-1]  
    y = dataset.iloc[:, -1]
    

    if request.method == 'POST':
        split_method = request.form.get('split_method')
        test_size = float(request.form.get('test_size', '0.2'))
        random_state = int(request.form.get('random_state', '42'))
        k_folds = int(request.form.get('k_folds', '5'))
        train_size = 1-test_size
        
        if split_method == 'test_train_split':
            x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=test_size, train_size=train_size, random_state=random_state)
            return render_template('split.html', split_method=split_method, test_size=test_size, train_size=train_size, random_state=random_state)
        elif split_method == 'k_fold':
            kf= KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
            for train_index, test_index in kf.split(dataset):
                x_train, x_test = dataset.iloc[train_index, :-1], dataset.iloc[test_index, :-1]
                y_train, y_test = dataset.iloc[train_index, -1], dataset.iloc[test_index, -1]
            return render_template('split.html', split_method=split_method, k_folds=k_folds, random_state=random_state)
        else:
            return "Invalid split method selected."

    # Handle other cases or render the form initially
    return render_template('split.html')
@app.route('/accuracy', methods=['POST', 'GET'])
def calc_accura():
    global dataset, x_train, x_test, y_train, y_test, model, current_model_file
    saved_models = []
    try:
        for file in os.listdir(app.root_path):
            if file.endswith('.pkl'):
                saved_models.append(file)
    except:
        return render_template("accuracy.html", accuracy="Hello")

    if request.method == 'POST':
        algorithm = request.form.get('algorithm')
        if algorithm == 'random_forest':
            model = RandomForestClassifier()
        elif algorithm == 'knn':
            model = KNeighborsClassifier()
        elif algorithm == 'svc':
            model = SVC()
        else:
            return render_template('accuracy.html', accuracy="Invalid Algorithm Selected")

        if x_train is None or y_train is None:
            return render_template('accuracy.html', accuracy="Dataset not split yet. Please split the dataset first.")

        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        now = datetime.now()
        current_time = now.strftime("%H-%M-%S--%d-%b-%Y")
        model_filename = '{}_model-{}.pkl'.format(algorithm, current_time)
        model_path = os.path.join(app.root_path, model_filename)
        save_model(model, model_path)
        current_model_file = model_filename

        return render_template('accuracy.html', accuracy=accuracy, saved_models=saved_models,
                               current_model_file=current_model_file)

    return render_template('accuracy.html', saved_models=saved_models, current_model_file=current_model_file) 

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global x_train, y_train, model, attributes, encoder, saved_models

    if x_train is None or y_train is None:
        return "Dataset not split yet. Please split the dataset first."

    if request.method == 'POST':
        print(request.form)  # Add this line to print the form data
        input_data = {}

        for feature in attributes:
            if feature != y_train.name:
                value = request.form.get(feature)
                input_data[feature] = [value]

        input_df = pd.DataFrame(input_data)

        # Handling missing values
        input_df.fillna(input_df.mean(), inplace=True)

        model_file = request.form['model']

        model_path = os.path.join(app.root_path, model_file)

        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        prediction = model.predict(input_df)
        if prediction == 1:
            prediction = "Water is Safe to drink"
        elif prediction==0:
            prediction = "Water is not safe to drink"
        else:
            prediction=prediction

        return render_template('predict.html', prediction=prediction, attributes=attributes, saved_models=saved_models)

    attributes = [col for col in dataset.columns if col != y_train.name] if dataset is not None else []
    try:
        saved_models = [file for file in os.listdir(app.root_path) if file.endswith('.pkl')]
    except:
        saved_models = []
    return render_template('predict.html', prediction=None, attributes=attributes, saved_models=saved_models)

@app.route('/compare', methods=['POST', 'GET'])
def compare_algorithms():
    if request.method == 'POST':
        selected_algorithms = request.form.getlist('algorithm')
        accuracies={}

        for algorithm in selected_algorithms:
            accuracy = calculate_accuracy(algorithm, x_train, x_test, y_train, y_test)
            accuracies[algorithm] = accuracy
        plt.bar(accuracies.keys(), accuracies.values())
        plt.xlabel('Algorithm')
        plt.ylabel('Accuracy')
        plt.title('Algorithm Comparison')
        plt.ylim([0, 1])

        plt.savefig('static/comparison_plot.png')
        return render_template('compare.html', accuracies=accuracies)
    else:
        return render_template('compare.html')



if __name__ == '__main__':
    app.run(debug=True)
