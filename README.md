# CMPT459-Milestone3

## Purpose

### data_preprocessing.py
The code in this file implements preprocessing of the data. In addition, new attrributes are calculated and addted to both training and testing dataset.
Running the code produces two JSON files in the zip format. These files are further used to perform train classifiers. The "new_test.json.zip" file is not included in this
repository as it exceeds maximum file size of Github (100 MB). 

1) new_train.json.zip
2) new_test.json.zip

### random_forest.py 
Trains and tests the Random Forest Classifier. Implemented by Kane.

### xgb.ipynb
Trains and tests the XGBoost algorithm. Implemented by Varun.

### stacking.py
Trains and tests the Stacking Classifier. Implemented by Anmol.

## Order of operation

The order of operation is relevant.
The code in "data_preprocessing.py" should be run first so that the classifiers are able to use generated files to train and test the models..

## Running Python Files


Running data_preprocessing.py
`python3 data_preprocessing.py`

Running random_forest.py
`python3 random_forest.py`

Running stacking.py
`python3 stacking.py`

Running xgb.ipynb
1. Start Jupyter Notebook server
`$ jupyter notebook`
2. Click on `xgb.ipynb` to open the file
3. Run the file by `Kernal->Restart & Run All`



## Libraries Used
- sklearn
- xgboost
- skimage
- numpy
- pandas
- seaborn
- scipy
- matplotlib
- sklearn
- nltk

