import pandas as pd
from sklearn.model_selection import train_test_split


class DataManager:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.filepath)
        print(f"Data loaded: {self.df.shape[0]} lignes, {self.df.shape[1]} colonnes")
        return self.df

    def preprocess_classification(self):
        self.df['sex'] = self.df['sex'].map({'female': 0, 'male': 1})
        self.df['smoker'] = self.df['smoker'].map({'no': 0, 'yes': 1})
        self.df = pd.get_dummies(self.df, columns=['region'], drop_first=True)
        features = self.df.drop(columns=['charges', 'smoker'])
        target = self.df['smoker']
        return features, target

    def preprocess_regression(self):
        df = self.df.copy()

        if df['sex'].dtype == 'object':
            df['sex'] = df['sex'].str.lower().str.strip()
            df['sex'] = df['sex'].map({'female': 0, 'male': 1})

        if df['smoker'].dtype == 'object':
            df['smoker'] = df['smoker'].str.lower().str.strip()
            df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})

        df = df.dropna()

        features = df.drop(columns=['smoker', 'charges'])
        target = df['charges']

        return features, target



    def preprocess_clustering(self):
        df = self.df.copy()

        if df['sex'].dtype == 'object':
            df['sex'] = df['sex'].str.lower().str.strip()
            df['sex'] = df['sex'].map({'female': 0, 'male': 1})
        if df['smoker'].dtype == 'object':
            df['smoker'] = df['smoker'].str.lower().str.strip()
            df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})

        if 'region' in df.columns:
            df = pd.get_dummies(df, columns=['region'], drop_first=True)

        return df.copy()

    def split_data(self, features, target, test_size=0.2, random_state=42):
        return train_test_split(features, target, test_size=test_size, random_state=random_state)
