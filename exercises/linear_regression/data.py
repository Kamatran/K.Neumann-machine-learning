from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import math
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
def load_frame():
    data = fetch_california_housing(download_if_missing=True, as_frame=True)
    return data.frame


def split_data(df, seed=42):
    train_val, test = train_test_split(df, test_size = 0.2, random_state=seed)
    train, val = train_test_split(train_val, test_size= 0.25, random_state=seed)
    return train, val, test

def add_region(df):
    out = df.copy()
    centers = {
        "LA": [34.05, -118.24],
        "SF": [37.77, -12.42],
        "SD": [32.72, -117.16],
        "SJ": [37.34, -121.89],
        "SAC": [38.58, -121.49]
    }

    regions = []
    for index, row in out[["Latitude", "Longitude"]].iterrows():
        latitude = row["Latitude"]
        longitude = row["Longitude"]

        min_dist = -1
        best_region = None
        for name, (center_latitude, center_longitude) in centers.items():
            dist = abs(math.dist([latitude, longitude], [center_latitude, center_longitude]))

            if  min_dist == -1 or dist < min_dist:
                min_dist = dist
                best_region = name

        if min_dist < 0.75:
            regions.append(best_region)
        else:
            regions.append("other")
    out["region"] = regions
    return out

def one_hot(train, other, column, drop_first=True):
    categories = sorted(train[column].unique())

    encoder = OneHotEncoder(
        categories=[categories],      
        drop='first' if drop_first else None,  
        sparse_output=False,          
        handle_unknown='ignore'       
    )
    
    # 3. Fit on train, transform both
    train_encoded = encoder.fit_transform(train[[column]])
    other_encoded = encoder.transform(other[[column]])

    new_columns = encoder.get_feature_names_out([column])
    
    train_encoded_df = pd.DataFrame(
        train_encoded, 
        columns=new_columns, 
        index=train.index
    )
    other_encoded_df = pd.DataFrame(
        other_encoded, 
        columns=new_columns, 
        index=other.index
    )

    train = train.drop(columns=[column])
    other = other.drop(columns=[column])
    
    train = pd.concat([train, train_encoded_df], axis=1)
    other = pd.concat([other, other_encoded_df], axis=1)

    return train, other, list(new_columns)

if __name__ == "__main__":

    df = load_frame()
    out = add_region(df)
    train, val, test = split_data(out)
    t, o, new = one_hot(train, val, "region", False)
    print(new)
    print(t["region_LA"].sum())
