

import pandas as pd
import numpy as np
import sys

from utils.metrics import calc_overall_metric

THRESHOLD = 0.5
PREDS_DIR = "preds_1dcnn"
if sys.argv[1] == 'final':
    PREDS_DIR = f'{PREDS_DIR}_ae_final'

type_to_fault = {
    "TYPE0":"all_normal",
    "TYPE1":"M1",
    "TYPE2":"M2",
    "TYPE3":"M3",
    "TYPE4":"M4",
    "TYPE5":"G1",
    "TYPE6":"G2",
    "TYPE7":"G3",
    "TYPE8":"G4",
    "TYPE9":"G5",
    "TYPE10":"G6",
    "TYPE11":"G7",
    "TYPE12":"G8",
    "TYPE13":"LA1",
    "TYPE14":"LA2",
    "TYPE15":"LA3",
    "TYPE16":"LA4",
    "TYPE17":"RA1"
}

motor_faults = ("M1", "M2", "M3", "M4")
gearbox_faults = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")
leftaxlebox_faults = ("LA1", "LA2", "LA3", "LA4")
rightaxlebox_faults = ("RA1")

# compile preds
for n in range(1,18):
    anomaly_type = f"TYPE{n}"
    if n==1:
        df = pd.read_csv(f"{PREDS_DIR}/preds_{type_to_fault[anomaly_type]}_1dcnn.csv")
        df = df[df.columns[0:2]]
        continue
    df_ = pd.read_csv(f"{PREDS_DIR}/preds_{type_to_fault[anomaly_type]}_1dcnn.csv")
    df = df.merge(df_[df_.columns[0:2]], on='sample')

# create labels from the compiled preds
for sample in range(1, len(df)+1):
    motor_string = 'M0'
    for fault in motor_faults:
        if df.loc[sample-1, fault] > THRESHOLD:
            if motor_string == 'M0':
                motor_string = f'{fault}'
                continue
            motor_string = motor_string + f'+{fault}'
        
    gearbox_string = '_G0'
    for fault in gearbox_faults:
        if df.loc[sample-1, fault] > THRESHOLD:
            if gearbox_string == '_G0':
                gearbox_string = f'_{fault}'
                continue
            gearbox_string = gearbox_string + f'+{fault}'

    leftaxlebox_string = '_LA0'
    for fault in leftaxlebox_faults:
        if df.loc[sample-1, fault] > THRESHOLD:
            if leftaxlebox_string == '_LA0':
                leftaxlebox_string = f'_{fault}'
                continue
            leftaxlebox_string = leftaxlebox_string + f'+{fault}'

    rightaxlebox_string = '_RA0'
    if df.loc[sample-1, "RA1"] > THRESHOLD:
        rightaxlebox_string = f'_RA1'

    df.loc[sample-1, "label"] = motor_string + gearbox_string + leftaxlebox_string + rightaxlebox_string

df2 = pd.read_csv('Data_Final_Stage/Data_Final_Stage/true_labels_round8.csv')
true_labels = pd.read_csv("Preliminary stage/Test_Labels_prelim.csv")
true_labels = true_labels["FaultCode"]
if sys.argv[1] == 'final':
    true_labels = df2[df2['New Sample']=='NO']['Label'].values.tolist()
pred_labels = df.loc[:len(true_labels)-1,"label"].values.tolist()
print(len(true_labels), len(pred_labels))
df = df[:len(true_labels)][['sample','label']].copy()
df['true_labels']= true_labels
df.to_csv("pred_labels.csv", index=False)
accuracy, precision, recall, f1, overall_z = calc_overall_metric(true_labels, pred_labels, test_set=bool(sys.argv[1] == 'test'))
print(f"{sys.argv[1]} | Acc={accuracy:.4f} | Pre={precision:.4f} | Rec={recall:.4f} | F1={f1:.4f} | Z={overall_z:.4f}")
