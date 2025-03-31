import pandas as pd
df = pd.read_csv('./Data/features_3_sec.csv')
print("All columns:", list(df.columns))
print("Total features for training:", len(df.columns) - 3)  # minus filename, length, label