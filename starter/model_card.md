# Model Card

This card is about a model trained on Adults dataset. Its main goal is to predict the salary of individuals
whether it is greater or less than 50K$.

## Model Details
Kaouther Bouzouita, by completing a project of MLOPs in udacity, created this model.
It is a Decision Tree classifier using default parameters of scikit-learn 1.3.0.
- Model date: July 20th, 2023
– Model version: 1.0.0
– Model type: Random Forest Classifier
– License: Creative Commons License

## Intended Use
This model should be used to predict individuals' salary, whether greater or less than 50000$,
given their personal data, such as education, number of working hours per week and native country.
The users are employment offices, job seekers and companies.

## Training Data
The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/20/census+income).
The original dataset has 32561 rows. An 80-20 split was used to break this info a train and test set. No stratification was done.
To use the data for training a One Hot Encoder was used on the categorical features and a label binarizer was used on the salary.

## Evaluation Data
After splitting the data, a 20% dataset was used for testing the trained model. Its preprocessing uses the same OneHotEncoder and Label Binarizer,
as the one trained during the training data preprocessing.

## Metrics

The model was evaluated using:
- Precision (with a value equal to 0.6373);
- Recall (with a value equal to 0.6365); and
- F-beta score, with beta equal to 1  (with a value equal to 0.637).

The model was also evaluated using Data Slicing, by measuring the different metrics on every value of a category, for categorical features.
The model has low scores for unknown workclass, which reflects the importance of such an attribute on the model training.


## Ethical Considerations
The number of males is almost the double of the females' number. But their respective scores are almost the same. 
This shows that the salary is not significantly influenced by gender.

## Caveats and Recommendations
There is an unbalance regarding native-country feature. The majority of individuals are
coming from the United States.
It is better to have more data about individuals from all countries. Native-countries data slicing will help us 
understand which variables are mre correlated to salary for each country.
