# Disaster Response Pipeline Project

### Project overview
The code in this produces uses a database of messages produced in
reponse to a disaster and their corresponding categories. This data is
cleansed and outputted to an sqlite database. This process is
performed in the data folder.

A classifier is then trained using the cleansed data in the sqlite
database and the optimal model is found using GridSearchCV. This model
is then saved using pickle and will be used for future predictions.
This process is performed in the models folder.

A Flask server is produced using the files in the app folder. The home
page has a few graphs which describe the dataset used for training and
validation. Furthermore it is possible to submit messages and get the
predicted categories of the  message returned as an output.

### Dataset issues
Note that the dataset is imbalanced and includes an even distribution of all the labels with some labels being absent such as the child_alone category.
The related category also had a 2 flag which likely means that it was ambiguous whether the message was related or not. Therefore the value of 2 was replaced with the mode which in this instance is 1 (a 1 means that the category is related to the message).

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### File description
```
.
├── app
│   ├── run.py
│   └── templates
│       ├── go.html
│       └── master.html
├── data
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   ├── DisasterResponse.db
│   ├── process_data.py
│   └── YourDatabaseName.db
├── models
│   ├── classifier.pkl
│   └── train_classifier.py
└── README.md
```
4 directories, 11 files
