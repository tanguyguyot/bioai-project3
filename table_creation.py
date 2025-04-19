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
import numpy as np

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
    
def to_binary_representation(tuples, length) -> str:
    binary_rep = "0" * length
    for i in tuples:
        binary_rep = binary_rep[:i] + "1" + binary_rep[i+1:]
    return binary_rep

def get_complete_table(error_table, length_columns, penalty_factor) -> dict:
    complete_table = {}
    for key, value in error_table.items():
        complete_table[key] = [to_binary_representation(key, length_columns), value, value - penalty_factor * len(key),
        ]
    complete_table["()"] = ["0" * length_columns, 0, 0]
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
    
def visualization_2d(complete_table: dict, dataset_name: str) -> None:
    # Add a column for convert binary representation to int
    complete_table['Binary_to_int'] = complete_table['Binary representation'].apply(lambda x: int(x, 2))
    X = complete_table['Binary_to_int']
    y = complete_table['Lookup value']
    ind=np.argsort(X)
    
    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5)
    plt.plot(X[ind], y[ind], color='red', linewidth=0.5)
    plt.title(f"Scatter plot of Lookup value vs Binary representation for {dataset_name}")
    plt.xlabel("Binary representation (as int)")
    plt.ylabel("Lookup value")
    plt.grid()
    plt.savefig(f"outputs/{dataset_name}_plot.png")
    plt.show()
    return
    """
def hinged_bitstring_map(complete_table:dict, dataset_name: str) -> None:
    # SPlit bitstrings into 2 parts
    half_length = (len(complete_table['Binary representation'][0]) // 2) + 1
    complete_table['Binary_part0'] = complete_table['Binary representation'].apply(lambda x: x[:half_length]).apply(lambda x: int(x, 2))
    complete_table['Binary_part1'] = complete_table['Binary representation'].apply(lambda x: x[half_length:]).apply(lambda x: int(x, 2))
    
    X = complete_table['Binary_part0']
    y = complete_table['Binary_part1']
    
    # Color gradient according to the score
    color_gradient = complete_table['Lookup value'].apply(lambda x: (x, 0.1, 0.1))
    
    plt.scatter(X, y)
    plt.title(f"HBM of {dataset_name}")
    plt.xlabel("Binary representation part 1")
    plt.ylabel("Binary representation part 2")
    plt.grid()
    plt.savefig(f"{dataset_name}_hinged_bitstring.png")
    plt.show()
    return"""
    
    
def hamming_distance(x, y):
    assert len(x) == len(y), "Strings must be of the same length"
    return sum(el1 != el2 for el1, el2 in zip(x, y))

    
if __name__ == "__main__":
    
    # Load complete table csv
    
    glass_complete_table = pd.read_csv("glass_complete_table.csv", dtype={'Binary representation': str})
    wine_complete_table = pd.read_csv("wine_complete_table.csv", dtype={'Binary representation': str})
    magic_complete_table = pd.read_csv("magic_complete_table.csv", dtype={'Binary representation': str})
    
    print(glass_complete_table)
    visualization_2d(glass_complete_table, "glass")
    visualization_2d(wine_complete_table, "wine")
    visualization_2d(magic_complete_table, "magic")
    
