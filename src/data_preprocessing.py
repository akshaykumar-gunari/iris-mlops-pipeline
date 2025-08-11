import pandas as pd
from sklearn.datasets import load_iris
import os

def preprocess_and_save(data_path='data/iris.csv'):
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df.to_csv(data_path, index=False)
    print(f"Iris dataset saved to {data_path}")

if __name__ == "__main__":
    preprocess_and_save()

