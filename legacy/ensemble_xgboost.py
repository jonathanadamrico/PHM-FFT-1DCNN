# ensemble_xgboost.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

preds_1dcnn = pd.read_csv("compiled_results_preds_1dcnn_final.csv")
preds_xgb = pd.read_csv("final_preds_df_xgboost_v10.csv")
preds_ensemble = preds_1dcnn.copy()

# Sort XGBoost preds according to sample
preds_xgb['sortby'] = -1
for i in range(len(preds_xgb)):
    preds_xgb.loc[i, 'sortby'] = int(preds_xgb.loc[i,'sample'][6:])
preds_xgb = preds_xgb.sort_values('sortby').reset_index(drop=True)

# Take average for G1, G8, LA2
preds_ensemble['G1'] = (preds_1dcnn['G1'] + preds_xgb['5']) / 2.0
preds_ensemble['G8'] = (preds_1dcnn['G8'] + preds_xgb['12']) / 2.0
preds_ensemble['LA2'] = (preds_1dcnn['LA2'] + preds_xgb['14']) / 2.0

# Save preds_ensemble
preds_ensemble.to_csv("compiled_results_preds_ensemble_final.csv", index=False)

print(preds_ensemble)
