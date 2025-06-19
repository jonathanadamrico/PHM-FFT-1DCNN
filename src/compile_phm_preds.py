# compile_phm_preds.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

RESULTS_DIR = "preds_1dcnn_final"

for i, csv_file in enumerate(os.listdir(RESULTS_DIR)):
    if not csv_file.startswith("pred"):
        continue
    if i == 0:
        df = pd.read_csv(f"{RESULTS_DIR}/{csv_file}")
        cols = df.columns
        if len(cols) == 3:
            df = df[[cols[0], cols[2], cols[1]]]
        continue
    df_ = pd.read_csv(f"{RESULTS_DIR}/{csv_file}")
    df = pd.concat([df, df_[df_.columns[1]]], axis=1)

df.to_csv(f"compiled_results_{RESULTS_DIR}.csv", index=False)

#cm = sns.light_palette("green", as_cmap=True)
#df.style.background_gradient(cmap=cm, axis=1)
#df.plot()
#plt.show()
