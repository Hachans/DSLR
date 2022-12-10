import sys
from LogisticRegression import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    test_data = pd.read_csv(sys.argv[1])
    model = LogisticRegression()
    
    model.load_values(sys.argv[2])
    test_data = test_data.fillna(method="ffill")
    X = np.array(test_data[['Defense Against the Dark Arts', 'Charms', 'Herbology', 'Ancient Runes', 'Muggle Studies']].values, dtype=float)
    sc = StandardScaler().fit(X)
    X_std = sc.transform(X)
    results = pd.DataFrame(model.predict(X_std))
    print(results.head())
    results.to_csv("data/houses.csv")