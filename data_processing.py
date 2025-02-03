import pandas as pd
import torch

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    columns = ['Timestamp', 'Speed', 'Course', 'Latitude', 'Longitude', 'Vessel']
    new_df = df[columns].copy()
    new_df.dropna(inplace=True)
    new_df.replace('masked', pd.NA, inplace=True)
    new_df.dropna(inplace=True)
    
    new_df['Latitude'] = new_df['Latitude'].astype(float)
    new_df['Longitude'] = new_df['Longitude'].astype(float)
    new_df['Speed'] = new_df['Speed'].astype(float)
    new_df['Course'] = new_df['Course'].astype(float)
    
    return new_df

def filter_data(df):
    df = df[(df['Longitude'] >= 32) & (df['Longitude'] <= 44)]
    df = df[(df['Latitude'] >= 12) & (df['Latitude'] <= 33)]
    df.sort_values(by=['Timestamp'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def filter_vessels(df, max_points=150):
    vessel_counts = df.groupby('Vessel').size()
    vessels_to_keep = vessel_counts[vessel_counts <= max_points].index
    return df[df['Vessel'].isin(vessels_to_keep)]

def get_data_statistics(df):
    stats = {
        'Min Latitude': df['Latitude'].min(),
        'Max Latitude': df['Latitude'].max(),
        'Min Longitude': df['Longitude'].min(),
        'Max Longitude': df['Longitude'].max(),
        'Min Timestamp': df['Timestamp'].min(),
        'Max Timestamp': df['Timestamp'].max(),
        'Min Speed': df['Speed'].min(),
        'Max Speed': df['Speed'].max(),
        'Min Course': df['Course'].min(),
        'Max Course': df['Course'].max(),
    }
    return stats

def main():
    path = '/content/DS-ALS-REDSEA-CLEANED.csv'
    df = load_data(path)
    df = preprocess_data(df)
    df = filter_data(df)
    df = filter_vessels(df)
    stats = get_data_statistics(df)
    
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    return df

if __name__ == "__main__":
    df_cleaned = main()
