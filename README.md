# Disaster Response Pipeline Project

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
