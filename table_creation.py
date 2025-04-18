# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import tree
from itertools import combinations
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt

def get_error_table(dataset, feature_columns, y, test_size=0.3, max_depth=3) -> dict:
    error_table = {}
    for i in tqdm(range(1, len(feature_columns) + 1)):
        for selected_columns in combinations(feature_columns, i):
            X = dataset[list(selected_columns)]
            # Get column indexes
            col_indexes = tuple([feature_columns.index(col) for col in selected_columns])
            # Split dataset into training set and test set
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
            
            # Create a clf
            clf = DecisionTreeClassifier(max_depth=max_depth, random_state=1)
            
            # Train clf
            clf = clf.fit(X_train, y_train)
            
            # Predict the response for test dataset
            y_pred = clf.predict(X_test)
            
            # Calculate fitness
            accuracy = metrics.accuracy_score(y_test, y_pred)
            error = 1 - accuracy
            error_table[col_indexes] = error
    return error_table

def get_lookup_table(error_table, penalty_factor):
    lookup_table = {}
    for key, value in error_table.items():
        lookup_table[key] = value - penalty_factor * len(key)
    return lookup_table

def print_results(error_table, lookup_table, dataset_name):
    print("Results for dataset: ", dataset_name)
    print(f"Lookup wine table:{lookup_table}")
    print(f"Error table wine: {error_table}")
    print(f"Length : {len(lookup_table)}")

    # Get the 10 max values and key of error table
    max_values = sorted(error_table.values(), reverse=True)[:10]
    max_keys = sorted(error_table, key=error_table.get, reverse=True)[:10]
    print(f"Max values glass (error table): {max_values}")
    print(f"Max keys glass (error table): {max_keys}")
    print(f"Max values glass (lookup table): {sorted(lookup_table.values(), reverse=True)[:10]}")
    print(f"Max keys glass (lookup table): {sorted(lookup_table, key=lookup_table.get, reverse=True)[:10]}")
    
def to_binary_representation(tuples, length):
    binary_rep = "0" * length
    for i in tuples:
        binary_rep = binary_rep[:i] + "1" + binary_rep[i+1:]
    return binary_rep

def get_complete_table(error_table, length_columns, penalty_factor):
    complete_table = {}
    for key, value in error_table.items():
        complete_table[key] = [to_binary_representation(key, length_columns), value, value - penalty_factor * len(key),
        ]
    return complete_table

def export_to_csv(complete_table, dataset_name):
    df = pd.DataFrame.from_dict(
    complete_table,
    orient='index',
    columns=['Binary representation', 'Error', 'Lookup value']
)
    # give name to index "features selected"
    df.index.name = "Features selected"
    df.to_csv(f'{dataset_name}.csv', index=True)
    print(f"Complete table exported to {dataset_name}.csv")
    
    
if __name__ == "__main__":
    print(to_binary_representation((1, 2), 5))