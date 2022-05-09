import keras
import pandas as pd
import streamlit as st

from titanicPreprocessing import preprocess
from utils import load_pickle_obj

st.title("Titanic Project")
st.write(
    "The sinking of the Titanic is one of the most infamous shipwrecks in history."
    "On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic"
    " sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard,"
    " resulting in the death of 1502 out of 2224 passengers and crew."
    "While there was some element of luck involved in surviving, it seems some groups of people were more "
    "likely to survive than others."
)

picklesc = load_pickle_obj('./save_best_model/pickle_sc')


st.markdown("### Enter the data and see if you would have survived the Titanic disaster or not")

name = st.text_input('Name', 'Giovanni Pinna')
ticket_class = st.selectbox('Ticket Class', ('1st', '2nd', '3rd'))
title_name = st.selectbox('Name title', ('Dr.', 'Master.', 'Mr.', 'Miss.', 'Mrs.', 'Reverend.', 'Other'))
sex = st.selectbox('Sex', ('male', 'female'))
sibsp = st.slider('Number of siblings / spouses aboard the Titanic', 0, 8)
parch = st.slider('Number of parents / children aboard the Titanic', 0, 6)
fare = st.slider('Passenger fare', 10.0, 600.0)
age = st.slider('Age in years', 0, 100)
cabin = st.selectbox('Cabin letter', ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'))
embarked = st.selectbox('Port of Embarkation', ('Cherbourg', 'Queenstown', 'Southampton'))

# process data
if ticket_class == '1st':
    class_number = 1
elif ticket_class == '2nd':
    class_number = 2
else:
    class_number = 3

if embarked == 'Cherbourg':
    embarked = 'C'
elif embarked == 'Queenstown':
    embarked = 'Q'
else:
    embarked = 'S'

if title_name == 'Other': title_name = 'Rare'

data = pd.DataFrame({
    'PassengerId': [0],
    'Survived': [0],
    'Pclass': [class_number],
    'Name': [title_name],
    'Sex': [sex],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Ticket': [0],
    'Fare': [fare],
    'Cabin': [cabin],
    'Embarked': [embarked]
})

p = preprocess(data)
p.set_sc_Age(picklesc[0])
p.set_sc_Fare(picklesc[1])
p.do_preprocess_for_line()
data_processed = p.get_data_train()
data_processed.drop(columns=['Survived'], inplace=True)

df = pd.DataFrame({
    'PassengerId': [0],
    'Pclass': [0],
    'Age': [0],
    'SibSp': [0],
    'Parch': [0],
    'Fare': [0],
    'cabin_multiple': [0],
    'Sex_female': [0],
    'Sex_male': [0],
    'Embarked_C': [0],
    'Embarked_Q': [0],
    'Embarked_S': [0],
    'cabin_letter_0': [0],
    'cabin_letter_A': [0],
    'cabin_letter_B': [0],
    'cabin_letter_C': [0],
    'cabin_letter_D': [0],
    'cabin_letter_E': [0],
    'cabin_letter_F': [0],
    'cabin_letter_G': [0],
    'cabin_letter_T': [0],
    'name_title_Dr': [0],
    'name_title_Master': [0],
    'name_title_Miss': [0],
    'name_title_Mr': [0],
    'name_title_Mrs': [0],
    'name_title_Rare': [0],
    'name_title_Rev': [0]
})

data_all = pd.concat([df, data_processed], ignore_index=True, sort=False)
data_all = data_all.fillna(0)
data_all = data_all.drop(0)

st.sidebar.markdown("# Summary : ")
for col in data_processed.loc[:, data_processed.columns != 'PassengerId'].columns:
    st.sidebar.write(f"{col} : {data_processed.iloc[0][col]}")

pickle = load_pickle_obj('./save_best_model/pickle_best_models')
svc_best = pickle[0]
xgb_best = pickle[1]
voting_clf_best = pickle[2]

model = keras.models.load_model("./save_best_model_pickle/keras_classifier3")

y_hat_svc_best = svc_best.predict(data_all.loc[:, data_all.columns != 'PassengerId']).astype(int)
y_hat_xgb_best = xgb_best.predict(data_all.loc[:, data_all.columns != 'PassengerId']).astype(int)
y_hat_voting_clf_best = voting_clf_best.predict(data_all.loc[:, data_all.columns != 'PassengerId']).astype(int)
y_hat_model = model.predict(data_all.loc[:, data_all.columns != 'PassengerId'])

results = dict([(1, 'Survived'), (0, 'Died')])

st.write(f"##### Dear {title_name} {name} the machine learning model says that : ")
st.write(f"##### Voting : {results.get(y_hat_voting_clf_best[0])}")
st.write(f"##### The deep learning model gives you this chance of survival : {y_hat_model[0]}")
st.write(f"SVM : {results.get(y_hat_svc_best[0])}")
st.write(f"XG boosting : {results.get(y_hat_xgb_best[0])}")

