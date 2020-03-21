from skimage.io import imread
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier



def k_neighbors_classifier(train_df, test_df):
    feature_cols = ['price', 'latitude', 'mean_des_tdidf', 
        'length_description', 'created_hour', 'closest_hospital',
        'closest_station', 'mean_feature_tdidf', 'created_day',
        'photos_num']

    x = train_df[feature_cols] # Features
    X_test = test_df[feature_cols]

    training_scores = []
    validation_scores = []
    validation_logloss = []

    knn = KNeighborsClassifier()

    targetMapping = {'high':0, 'medium':1, 'low':2}
    y= np.array(train_df['interest_level'].apply(lambda x: targetMapping[x]))

    #cross validation 
    kf = KFold(n_splits=5, shuffle = False)
    X = np.array(x)
    # y = np.array(y)
    for train_index, test_index in kf.split(X):
        X_train, X_validation = X[train_index], X[test_index]
        y_train, y_validation = y[train_index], y[test_index]
        k_nearest = knn.fit(X_train,y_train)
        y_pred = k_nearest.predict(X_validation)
        y_pred1 = k_nearest.predict_proba(X_validation)
        training_scores.append(k_nearest.score(X_train, y_train))
        validation_scores.append(accuracy_score(y_validation, y_pred))
        validation_logloss.append(log_loss(y_validation, y_pred1))

    #check overfitting and performance
    print('Scores from each Iteration: ', training_scores)
    print('Scores from each Iteration: ', validation_scores)
    print('Average k-fold on training: ', np.mean(training_scores))
    print('Average k-fold on testing: ', np.mean(validation_scores))
    print('Average k-fold on validation using logloss: ', np.mean(validation_logloss))

    #train classifier
    random_forest = k_nearest.fit(x,y)

    y_pred = k_nearest.predict_proba(X_test)

    submission = pd.DataFrame({
        "listing_id": test_df["listing_id"],
        "high": y_pred[:,0],
        "medium":y_pred[:,1],
        "low":y_pred[:,2]
    })

    titles_columns=["listing_id","high","medium","low"]
    submission=submission.reindex(columns=titles_columns)
    submission.to_csv('initial_k_nearest_submission.csv', index=False)

# Helpful resource:
# https://www.ritchieng.com/machine-learning-efficiently-search-tuning-param/
def parameter_tuning(train_df):
    feature_cols = ['bedrooms','bathrooms','price', 'latitude', 'mean_des_tdidf', 
    'length_description', 'created_hour', 'closest_station', 'closest_hospital', 
    'mean_feature_tdidf', 'created_day','photos_num']

    X = train_df[feature_cols]
    y = train_df['interest_level']

    knn = KNeighborsClassifier()

    # k_range = list(range(1, 31))
    k_range = []
    for i in range(100,501):
        if i%5 == 0:
            k_range.append(i)
    print(k_range)
    # k_range = list(range(100, 500))
    weight_options = ['uniform', 'distance']
    param_grid = dict(n_neighbors=k_range, weights=weight_options)
    print(param_grid)
    grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
    grid.fit(X, y)

    print(grid.best_score_)
    print(grid.best_params_)

def improved_k_neighbors_classifier(train_df, test_df):
    feature_cols = ['bedrooms', 'bathrooms', 'latitude', 
        'price', 'number_features',  'length_description', 'closest_station', 'closest_hospital',
        'created_month', 'created_day', 'created_hour', 'photos_num', 
        'mean_des_tdidf', 'mean_feature_tdidf']

    x = train_df[feature_cols] # Features
    X_test = test_df[feature_cols]

    training_scores = []
    validation_scores = []
    validation_logloss = []

    knn = KNeighborsClassifier(n_neighbors=500, weights='distance')

    targetMapping = {'high':0, 'medium':1, 'low':2}
    y= np.array(train_df['interest_level'].apply(lambda x: targetMapping[x]))

    #cross validation 
    kf = KFold(n_splits=5, shuffle = False)
    X = np.array(x)
    # y = np.array(y)
    for train_index, test_index in kf.split(X):
        X_train, X_validation = X[train_index], X[test_index]
        y_train, y_validation = y[train_index], y[test_index]
        k_nearest = knn.fit(X_train,y_train)
        y_pred = k_nearest.predict(X_validation)
        y_pred1 = k_nearest.predict_proba(X_validation)
        training_scores.append(k_nearest.score(X_train, y_train))
        validation_scores.append(accuracy_score(y_validation, y_pred))
        validation_logloss.append(log_loss(y_validation, y_pred1))

    #check overfitting and performance
    print('Improved Scores from each Iteration: ', training_scores)
    print('Improved Scores from each Iteration: ', validation_scores)
    print('Improved Average k-fold on training: ', np.mean(training_scores))
    print('Improved Average k-fold on testing: ', np.mean(validation_scores))
    print('Improved Average k-fold on validation using logloss: ', np.mean(validation_logloss))

    #train classifier
    random_forest = k_nearest.fit(x,y)

    y_pred = k_nearest.predict_proba(X_test)

    submission = pd.DataFrame({
        "listing_id": test_df["listing_id"],
        "high": y_pred[:,0],
        "medium":y_pred[:,1],
        "low":y_pred[:,2]
    })

    titles_columns=["listing_id","high","medium","low"]
    submission=submission.reindex(columns=titles_columns)
    submission.to_csv('improved_k_nearest_submission.csv', index=False)

def main():

    train_df = pd.read_json('new_train.json.zip')
    test_df = pd.read_json('new_test.json.zip')

    # print(train_df)
    # print(test_df)

    # parameter_tuning(train_df)
    # No such thing as feature selection in KNN 
  
    k_neighbors_classifier(train_df, test_df)
    print("-------------")
    improved_k_neighbors_classifier(train_df, test_df)






if __name__ == "__main__":
    main()