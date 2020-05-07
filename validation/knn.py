from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import sys
import time
import os

# Load results data
filename = sys.argv[1]
results_data = pd.read_csv(filename)
# Separate features/target


# Setup estimator
neigh = KNeighborsClassifier(n_neighbors=1)

# Run validation
start = time.time()
# accs = cross_val_score(neigh, X_new, y) # 5-fold cv
elapsed = time.time() - start
print('Completed validation in {} seconds.'.format(elapsed))

# Write results to CSV
# results_data['Accuracy_KNN'] = accs
results_data.to_csv('results/{}'.format(os.path.basename(filename)),\
    index=False)
print(results_data)
