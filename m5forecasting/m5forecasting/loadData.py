import pandas as pd
import numpy as np
import os

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "data")

df = pd.read_csv(os.path.join(DATA_PATH, "sales_train_validation.csv"))

bare_df = df.iloc[:,6:]

all_data = np.array(bare_df)

full_df = pd.read_csv(os.path.join(DATA_PATH, "sales_train_evaluation.csv"))

bare_revenue = pd.read_csv(os.path.join(DATA_PATH, "revenue.csv"))

calendar = pd.read_csv(os.path.join(DATA_PATH, "calendar.csv"))

def clean(df):
    return np.array(df.iloc[:30490, -28:])

actuals = clean(full_df)
