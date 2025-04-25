# Load libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, DecisionTreeClassifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from itertools import combinations
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def get_table(dataset, feature_columns, y, penalty_factor=0.01, test_size=0.3, cl="rf", max_depth=3) -> dict:
    error_table = {}
    for i in tqdm(range(1, len(feature_columns) + 1)):
        for selected_columns in combinations(feature_columns, i):
            X = dataset[list(selected_columns)]
            # Get column indexes
            col_indexes = tuple([feature_columns.index(col) for col in selected_columns])
            # Split dataset into training set and test set
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
            
            # Create a clf, specified by the step 6
            if cl == "rf":
                clf = RandomForestClassifier(n_estimators=30, criterion='gini', max_depth=None, min_samples_split=2, min_impurity_decrease=0.0, max_features="sqrt", n_jobs=4)
            else:
                clf = DecisionTreeClassifier(max_depth=max_depth, random_state=1)
            # Train clf
            clf = clf.fit(X_train, y_train)
            
            # Predict the response for test dataset
            y_pred = clf.predict(X_test)
            
            # Calculate fitness
            accuracy = metrics.accuracy_score(y_test, y_pred)
            error = 1 - accuracy # the lower the best
            binary_representation = to_binary_representation(col_indexes, len(feature_columns))
            error_table[binary_representation] = {
                "Error": error,
                "Features selected": col_indexes,
                "Lookup value": error + len(selected_columns) * penalty_factor,
                "Penalty factor": penalty_factor
            }
            
    # Add the empty individual
    error_table["0" * len(feature_columns)] =  {
        "Error": 1,
        "Features selected": tuple(),
        "Lookup value": 1,
        "Penalty factor": penalty_factor
    }
    
    return error_table

def change_penalty(table: dict, new_penalty: float) -> dict:
    """
    Change the penalty factor of the table
    """
    for key, value in table.items():
        table[key]["Penalty factor"] = new_penalty
        table[key]["Lookup value"] = value["Error"] + len(value["Features selected"]) * new_penalty
    return table

def to_binary_representation(tuples, length) -> str:
    """
    Convert a tuple of indexes to a binary representation
    """
    binary_rep = "0" * length
    for i in tuples:
        binary_rep = binary_rep[:i] + "1" + binary_rep[i+1:]
    return binary_rep

def is_local_minimum(bitstring: str, table: dict, lookup: False) -> bool:
    """
    Check if the bitstring is a local minimum in the error table.
    """
    length_columns = bitstring.count("1")
    # Get the error of the current bitstring
    current_error = table[bitstring]["Error"]
    if lookup:
        current_error = table[bitstring]["Lookup value"]
    
    # Check all neighbors
    for i in range(length_columns):
        neighbor = list(bitstring)
        neighbor[i] = "1" if neighbor[i] == "0" else "0"
        neighbor = "".join(neighbor)
        
        # If any neighbor has a lower error, return False
        if lookup:
            if table[neighbor]["Lookup value"] < current_error:
                return False
        else:
            if table[neighbor]["Error"] < current_error:
                return False
    return True

def get_local_minimums(table: dict, lookup: False) -> list:
    """
    Get all local minimums in the error table.
    """
    local_minimums = []
    for bitstring in table:
        if is_local_minimum(bitstring, table, lookup):
            if bitstring.count("1") == 0: # we don't want to display the empty individual
                continue
            if lookup:
                local_minimums.append((bitstring, int(bitstring, 2), table[bitstring]["Lookup value"]))
            else:
                local_minimums.append((bitstring, int(bitstring, 2), table[bitstring]["Error"]))
    # Sort the local minimums by error : global minimum is the first one
    local_minimums.sort(key=lambda x: x[2])
    return local_minimums

def export_to_csv(complete_table: dict, dataset_name: str) -> None:
    """Convert the dict to a csv file

    Args:
        complete_table (_type_): _description_
        dataset_name (_type_): _description_
    """
    table = {}
    for key, value in complete_table.items():
        table[key] = [value["Error"], value["Features selected"], value["Lookup value"], value["Penalty factor"]]
    complete_table = pd.DataFrame.from_dict(complete_table, orient='index', columns=["Error", "Features selected", "Lookup value", "Penalty factor"])
    complete_table.index.name = "Binary representation"
    complete_table.to_csv(f"outputs/{dataset_name}_complete_table.csv", index=True)
    print(f"Complete table saved to outputs/{dataset_name}_complete_table.csv")
    return
    
def csv_to_dict(csv_file: str) -> dict:
    """
    Convert a csv file to a dict
    """
    table = pd.read_csv(csv_file, dtype={'Binary representation': str})
    table.index = table['Binary representation']
    table = table.drop(columns=["Binary representation"])
    return table.to_dict(orient='index')

def visualization_2d(complete_table: dict, dataset_name: str) -> None:
    plt.close()
    X = []
    y = []
    for key, value in tqdm(complete_table.items()):
        # not displaying the empty individual
        if key.count("1") == 0:
            continue
        X.append(int(key, 2))
        y.append(value["Lookup value"])
    X = np.array(X)
    y = np.array(y)
    ind = np.argsort(X)
    
    # Display the local minimums on the plot
    local_minimums = get_local_minimums(complete_table, lookup=True)
    X_locals = [x[1] for x in local_minimums]
    y_locals = [x[2] for x in local_minimums]
    
    plt.plot(X[ind], y[ind], color='gray', linewidth=0.5, alpha=0.8)
    plt.scatter(X, y, alpha=0.5, color="green", s=10)
    plt.scatter(X_locals, y_locals, label='Local optimums', facecolors='none', edgecolors='blue')
    plt.scatter(X_locals[0], y_locals[0], label=f"Global minimum: {y_locals[0]:.4f}", facecolors='none', edgecolors='red')

    
    # Labels
    plt.title(f"Scatter plot of Lookup value vs Binary representation for {dataset_name}")
    plt.xlabel("Binary representation (as int)")
    plt.ylabel("Lookup value")
    plt.grid()
    plt.legend()
    plt.savefig(f"outputs/plots/{dataset_name}_plot.png")
    plt.show()
    return
    
def hinged_bitstring_map(complete_table: dict, dataset_name: str) -> None:
    plt.close()
        
    # Split bitstrings into 2 parts
    sample_key = next(iter(complete_table.keys()))
    length = len(sample_key)
    half_length = length // 2
    X = []
    y = []
    lookup_values = []
    for key, value in tqdm(complete_table.items()):
        X.append(int(key[:half_length], 2))
        y.append(int(key[half_length:], 2))
        lookup_values.append(value["Lookup value"])
    X = np.array(X)
    y = np.array(y)
    lookup_values = np.array(lookup_values)
    
    # Dégradé vert → beige → violet
    def color_value(x):
        start_color = np.array([0, 100, 0]) / 255        
        mid_color = np.array([255, 245, 245]) / 255      
        end_color = np.array([128, 0, 80]) / 255         

        if x <= 0.5:
            t = x / 0.5
            color = (1 - t) * start_color + t * mid_color
        else:
            t = (x - 0.5) / 0.5
            color = (1 - t) * mid_color + t * end_color
        return color

    # Couleurs pour chaque point
    color_gradient = [color_value(x) for x in lookup_values]

    # Générer une colormap pour la colorbar
    n_colors = 256
    cmap = ListedColormap([color_value(x) for x in np.linspace(0, 1, n_colors)])
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])

    # Préparer la figure
    fig, ax = plt.subplots(figsize=(length - 2, length))

    # Scatter des points
    sc = ax.scatter(X, y, c=color_gradient)

    # Points minimaux
    local_minimums = get_local_minimums(complete_table, lookup=True)
    X_locals = [int(x[0][:half_length], 2) for x in local_minimums]
    y_locals = [int(x[0][half_length:], 2) for x in local_minimums]
    X_locals = np.array(X_locals)
    y_locals = np.array(y_locals)

    # Ajouter global minimum
    global_minimum = local_minimums[0]
    X_locals = np.append(X_locals, int(global_minimum[0][:half_length], 2))
    y_locals = np.append(y_locals, int(global_minimum[0][half_length:], 2))

    # Marquer les minima
    ax.scatter(X_locals, y_locals, label='Local optimums', facecolors='none', edgecolors='blue')
    ax.scatter(X_locals[-1], y_locals[-1], label=f"Global minimum: {global_minimum[2]:.4f}", 
            s=50, facecolors='none', edgecolors='red')

    # Titres et légendes
    plt.title(f"HBM of {dataset_name}")
    plt.xlabel("Binary representation part 1")
    plt.ylabel("Binary representation part 2")
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Ajouter la colorbar
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1, aspect=40, shrink=0.8)
    cbar.set_label("Lookup value", fontsize=10)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.ax.set_xticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])

    # Sauvegarde et affichage
    plt.tight_layout()
    plt.savefig(f"outputs/plots/{dataset_name}_hbm.png")
    plt.show()
    return

def hamming_distance(x, y):
    assert len(x) == len(y), "Strings must be of the same length"
    return sum(el1 != el2 for el1, el2 in zip(x, y))

if __name__ == "__main__":
    
    # Load complete table csv
    glass_complete_table = csv_to_dict("outputs/glass_complete_table.csv")
    wine_complete_table = csv_to_dict("outputs/wine_complete_table.csv")
    magic_complete_table = csv_to_dict("outputs/magic_complete_table.csv")
    
    # print(glass_complete_table)
    visualization_2d(glass_complete_table, "glass")
    hinged_bitstring_map(glass_complete_table, "glass")
    
    visualization_2d(wine_complete_table, "wine")
    hinged_bitstring_map(wine_complete_table, "wine")
    
    visualization_2d(magic_complete_table, "magic")
    hinged_bitstring_map(magic_complete_table, "magic")
    
    
    
    
