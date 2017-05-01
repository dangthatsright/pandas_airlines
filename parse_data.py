import pandas as pd

allflights = pd.read_csv("domesticflights15.csv")

def clean_data(df):
    df = df[df['SEATS'] > 0]
    df = df[df['PASSENGERS'] > 0]
    return df


cleaned_df = clean_data(allflights)
cleaned_df.to_csv('cleaned_flights.csv')
