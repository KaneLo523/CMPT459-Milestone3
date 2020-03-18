from skimage.io import imread
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.metrics import accuracy_score #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import log_loss
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def random_forest_classifier(train_df, test_df):
    feature_cols = ['price', 'latitude', 'mean_des_tdidf', 
            'length_description', 'created_hour', 'closest_hospital',
            'closest_station', 'mean_feature_tdidf', 'created_day',
            'photos_num']
    x = train_df[feature_cols] # Features
    X_test = test_df[feature_cols]

    training_scores = []
    validation_scores = []
    validation_logloss = []

    random_forest = RandomForestClassifier()

    targetMapping = {'high':0, 'medium':1, 'low':2}
    y= np.array(train_df['interest_level'].apply(lambda x: targetMapping[x]))
    
    #cross validation 
    kf = KFold(n_splits=5, shuffle = False)
    X = np.array(x)
    # y = np.array(y)
    for train_index, test_index in kf.split(X):
        X_train, X_validation = X[train_index], X[test_index]
        y_train, y_validation = y[train_index], y[test_index]
        random_forest = random_forest.fit(X_train,y_train)
        y_pred = random_forest.predict(X_validation)
        y_pred1 = random_forest.predict_proba(X_validation)
        training_scores.append(random_forest.score(X_train, y_train))
        validation_scores.append(accuracy_score(y_validation, y_pred))
        validation_logloss.append(log_loss(y_validation, y_pred1))

    #check overfitting and performance
    print('Scores from each Iteration: ', training_scores)
    print('Scores from each Iteration: ', validation_scores)
    print('Average k-fold on training: ', np.mean(training_scores))
    print('Average k-fold on testing: ', np.mean(validation_scores))
    print('Average k-fold on validation using logloss: ', np.mean(validation_logloss))

    #train classifier
    random_forest = random_forest.fit(x,y)

    y_pred = random_forest.predict_proba(X_test)

    submission = pd.DataFrame({
        "listing_id": test_df["listing_id"],
        "high": y_pred[:,0],
        "medium":y_pred[:,1],
        "low":y_pred[:,2]
    })

    titles_columns=["listing_id","high","medium","low"]
    submission=submission.reindex(columns=titles_columns)
    submission.to_csv('initial_random_forest_submission.csv', index=False)

def improved_random_forest_classifier(train_df, test_df):
    feature_cols = ['bedrooms','bathrooms','price', 'latitude', 'mean_des_tdidf', 'length_description', 'created_hour', 'closest_station', 'closest_hospital', 'mean_feature_tdidf', 'created_day','photos_num']
    x = train_df[feature_cols] # Features
    X_test = test_df[feature_cols]

     # Parameter Tuning
    # max_depth = [3,5,3.5, 4, 4.5,5,5.5, 6, 6.5,7,8,9,10]
    # min_samples_split = [5,10,15,20,25,30,35,40]
    # n_estimators = [1,2,3,4,5,6,7,8,9,10,15]
    # parameters = [{'max_depth':max_depth,'min_samples_split':min_samples_split,'n_estimators':n_estimators}]
    # grid_search = GridSearchCV(estimator=random_forest, param_grid=parameters, scoring ='accuracy',cv=5,n_jobs=-1)
    # grid_search = grid_search.fit(x,y)

    # best_accuracy = grid_search.best_score_
    # print(best_accuracy)
    # # best_accuracy
    # opt_param = grid_search.best_params_
    # print ("The best: ", opt_param )

    training_scores = []
    validation_scores = []
    validation_logloss = []
    random_forest = RandomForestClassifier(n_estimators=1000, max_depth = 13)
    targetMapping = {'high':0, 'medium':1, 'low':2}
    y= np.array(train_df['interest_level'].apply(lambda x: targetMapping[x]))

    #cross validation 
    kf = KFold(n_splits=5, shuffle = False)
    X = np.array(x)
    # y = np.array(y)
    for train_index, test_index in kf.split(X):
        X_train, X_validation = X[train_index], X[test_index]
        y_train, y_validation = y[train_index], y[test_index]
        random_forest = random_forest.fit(X_train,y_train)
        y_pred = random_forest.predict(X_validation)
        y_pred1 = random_forest.predict_proba(X_validation)
        training_scores.append(random_forest.score(X_train, y_train))
        validation_scores.append(accuracy_score(y_validation, y_pred))
        validation_logloss.append(log_loss(y_validation, y_pred1))
    
    #check overfitting
    print('Scores from each Iteration: ', training_scores)
    print('Scores from each Iteration: ', validation_scores)
    print('Improved Average k-fold on training: ', np.mean(training_scores))
    print('Improved Average k-fold on validation: ', np.mean(validation_scores))
    print('Improved Average k-fold on validation using logloss: ', np.mean(validation_logloss))

    #retrain classifier on the whole dataset
    random_forest = random_forest.fit(X,y)

    y_pred = random_forest.predict_proba(X_test)

    submission = pd.DataFrame({
        "listing_id": test_df["listing_id"],
        "high": y_pred[:,0],
        "medium":y_pred[:,1],
        "low":y_pred[:,2]
    })

    titles_columns=["listing_id","high","medium","low"]
    submission=submission.reindex(columns=titles_columns)
    submission.to_csv('improved_random_forest_submission.csv', index=False)

   
def feature_Selection(train_df):
    cols = ['bedrooms', 'bathrooms', 'latitude', 
        'price', 'number_features', 
        'length_description', 'closest_station', 'closest_hospital',
        'created_month', 'created_day', 'created_hour', 'photos_num', 
        'mean_des_tdidf', 'mean_feature_tdidf']

    X = train_df[cols]
    y = train_df['interest_level']
    # Performing feature selection
    model = ExtraTreesClassifier()
    model.fit(X,y)
    #print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    # plt.show()
    #print(feat_importances.nlargest(10))

def new_feature_Selection(train_df):
    cols = ['bedrooms', 'bathrooms', 'latitude', 
        'price', 'number_features', 
        'length_description', 'closest_station', 'closest_hospital',
        'created_month', 'created_day', 'created_hour', 'photos_num', 
        'mean_des_tdidf', 'mean_feature_tdidf']

    X = train_df[cols]
    y = train_df['interest_level']
    # Performing feature selection
    model = ExtraTreesClassifier()
    model.fit(X,y)
    #print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(14).plot(kind='barh')
    # plt.show()
    # print(feat_importances.nlargest(14))



def main():

    train_df = pd.read_json('new_train.json.zip')
    test_df = pd.read_json('new_test.json.zip')

    # feature_Selection(train_df)
    random_forest_classifier(train_df, test_df)

    # new_feature_Selection(train_df)
    improved_random_forest_classifier(train_df, test_df)




if __name__ == "__main__":
    main()