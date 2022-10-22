# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the csv file
df = pd.read_csv('data_model.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
cat_df = df[['CODE_GENDER', 'EMERGENCYSTATE_MODE', 'OCCUPATION_TYPE',
             'WALLSMATERIAL_MODE']]
cat_df = pd.get_dummies(cat_df, drop_first=True)
num_df = df[['EXT_SOURCE_3', 'REGION_RATING_CLIENT', 'AMT_GOODS_PRICE',
             'GOODS_PRICE_CREDIT_PER', 'DAYS_WORKING_PER', 'ANNUITY_DAYS_BIRTH_PERC',
             'TARGET']]
data_encoded = pd.concat([cat_df, num_df], axis=1)

# Select independent and dependent variable
X = data_encoded.drop(['TARGET'], axis=1)
y = data_encoded['TARGET']

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Instantiate the model
classifier = GradientBoostingClassifier(learning_rate=0.017794191838477483, n_estimators=600,
                                        subsample=0.6159921598038572)

# Fit the model
classifier.fit(X_train, y_train)


# Make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))
