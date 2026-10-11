from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def load_frame():
    data = fetch_california_housing(download_if_missing=True, as_frame=True)
    return data.frame


def split_data(df, seed=42):
    train_val, test = train_test_split(df, test_size = 0.2, random_state=seed)
    train, val = train_test_split(train_val, test_size= 0.25, random_state=seed)
    return train, val, test

df = load_frame()
train, val, test = split_data(df)
