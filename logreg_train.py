import numpy as np
import pandas as pd
from LogisticRegression import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])
    df = df.dropna(subset=['Defense Against the Dark Arts'])
    df = df.dropna(subset=['Charms'])
    df = df.dropna(subset=['Herbology'])
    df = df.dropna(subset=['Ancient Runes'])
    df = df.dropna(subset=['Muggle Studies'])
    X = np.array(df[['Defense Against the Dark Arts', 'Charms', 'Herbology', 'Ancient Runes', 'Muggle Studies']].values, dtype=float)
    y = df.values[:, 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

    sc = StandardScaler()
    sc.fit(X_train)

    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    model = LogisticRegression(lrate=0.01, epochs=50).fit(X_train_std, y_train)
    model.save_values()