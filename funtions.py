import pandas as pd
import pickle
import joblib


#load the model
with open('RF_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

#load preprocessor
loaded_preprocessor = joblib.load('preprocessor.pkl')


value_mappings = {
    'q1': {
        0: "As much as I always could",
        1: "Not quite so much now",
        2: "Definitely not so much now",
        3: "Not at all"
    },
    'q2': {
        0: "As much as I ever did",
        1: "Rather less than I used to",
        2: "Definitely less than I used to",
        3: "Hardly at all"
    },
    'q3': {
        0: "Yes, most of the time",
        1: "Yes, some of the time",
        2: "Not very often",
        3: "No, never"
    },
    'q4': {
        0: "No, not at all",
        1: "Hardly ever",
        2: "Yes, sometimes",
        3: "Yes, very often"
    },
    'q5': {
        0: "Yes, quite a lot",
        1: "Yes, sometimes",
        2: "No, not much",
        3: "No, not at all"
    },
    'q6': {
        0: "Yes, most of the time I haven't been able to cope at all",
        1: "Yes, sometimes I haven't been coping as well as usual",
        2: "No, most of the time I have coped quite well",
        3: "No, I have been coping as well as ever"
    },
    'q7': {
        0: "Yes, most of the time",
        1: "Yes, sometimes",
        2: "Not very often",
        3: "No, not at all"
    },
    'q8': {
        0: "Yes, most of the time",
        1: "Yes, quite often",
        2: "Not very often",
        3: "No, not at all"
    },
    'q9': {
        0: "Yes, most of the time",
        1: "Yes, quite often",
        2: "Only occasionally",
        3: "No, never"
    },
    'q10': {
        0: "Yes, quite often",
        1: "Sometimes",
        2: "Hardly ever",
        3: "Never"
    }
}

score_mapping = [
    [0, 1, 2, 3],  # Question 1
    [0, 1, 2, 3],  # Question 2
    [3, 2, 1, 0],  # Question 3
    [0, 1, 2, 3],  # Question 4
    [3, 2, 1, 0],  # Question 5
    [3, 2, 1, 0],  # Question 6
    [3, 2, 1, 0],  # Question 7
    [3, 2, 1, 0],  # Question 8
    [3, 2, 1, 0],  # Question 9
    [3, 2, 1, 0],  # Question 10
]

new_order = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10',
             'Total Score', 'Age', 'Partner Support', 'Marital Status',
             'Sleep Quality', 'Fertility Treatment History', 'Diabetes Level']


#calculate score for given answer and create total score colum
def calculate_score(row, score_mapping):
    total_score = 0
    for i, mapping in enumerate(score_mapping):
        score_dict = dict(enumerate(mapping))
        total_score += score_dict[row[f'q{i+1}']]
    return total_score

def add_total_score(df, score_mapping):
    df['Total Score'] = df.apply(lambda row: calculate_score(row, score_mapping), axis=1)
    return df

#create data frame for given value
def create_df(values):
    df =pd.DataFrame([values], columns=['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10',
            'Age', 'Partner Support', 'Marital Status',
        'Sleep Quality', 'Fertility Treatment History', 'Diabetes Level'])
    
    df = add_total_score(df, score_mapping)
    
    df = df[new_order]

    for question, mapping in value_mappings.items():
        df[question] = df[question].replace(mapping)
    
    return df


#to predict stress category
def get_prediction(df):
    x =loaded_preprocessor.transform(df) 
    y_pred = loaded_model.predict(x)

    return y_pred