# create_submission_file.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator

df = pd.read_csv("compiled_results_preds_ensemble_final.csv")

THRESHOLD = 0.50

faults = [
    "M1",
    "M2",
    "M3",
    "M4",
    "G1",
    "G2",
    "G3",
    "G4",
    "G5",
    "G6",
    "G7",
    "G8",
    "LA1",
    "LA2",
    "LA3",
    "LA4"
]

motor_faults = ("M1", "M2", "M3", "M4")
gearbox_faults = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")
leftaxlebox_faults = ("LA1", "LA2", "LA3", "LA4")
rightaxlebox_faults = ("RA1")

submission_df = pd.DataFrame({"Sample Number": df["sample"]})
submission_df["motor"] = ''
submission_df["gearbox"] = ''
submission_df["leftaxlebox"] = ''
submission_df["rightaxlebox"] = ''
submission_df["Label"] = ''
print(submission_df.columns)

for sample in range(1, len(df)+1):
    motor_string = 'M0'
    for fault in motor_faults:
        if df.loc[sample-1, fault] > THRESHOLD:
            if motor_string == 'M0':
                motor_string = f'{fault}'
                continue
            
            # if 2nd to the last letter is equal to the second to the last letter
            #if motor_string[-2] == fault[-2]:
            motor_string = motor_string + f'+{fault}'
                
    # if component signals anomaly yet none is, get the one with highest confidence
    if motor_string == 'M0' and df.loc[sample-1, 'motor'] > THRESHOLD:
        d = df.loc[sample-1,["M1","M2","M3","M4"]].to_dict()
        highest_conf_fault = max(d.items(), key=operator.itemgetter(1))[0]
        motor_string = highest_conf_fault
        print(sample, highest_conf_fault)


    gearbox_string = '_G0'
    for fault in gearbox_faults:
        if df.loc[sample-1, fault] > THRESHOLD:
            if gearbox_string == '_G0':
                gearbox_string = f'_{fault}'
                continue
            
            # if 2nd to the last letter is equal to the second to the last letter
            #if gearbox_string[-2] == fault[-2]:
            gearbox_string = gearbox_string + f'+{fault}'
    
    # if component signals anomaly yet none is, get the one with highest confidence
    if gearbox_string == '_G0' and df.loc[sample-1, 'gearbox'] > THRESHOLD:
        d = df.loc[sample-1,list(gearbox_faults)].to_dict()
        highest_conf_fault = max(d.items(), key=operator.itemgetter(1))[0]
        gearbox_string = f'_{highest_conf_fault}'
        print(sample, highest_conf_fault)

    leftaxlebox_string = '_LA0'
    for fault in leftaxlebox_faults:
        if df.loc[sample-1, fault] > THRESHOLD:
            if leftaxlebox_string == '_LA0':
                leftaxlebox_string = f'_{fault}'
                continue
            
            # if 2nd to the last letter is equal to the second to the last letter "A"
            #if leftaxlebox_string[-2] == fault[-2]:
            leftaxlebox_string = leftaxlebox_string + f'+{fault}'

    # if component signals anomaly yet none is, get the one with highest confidence
    if leftaxlebox_string == '_LA0' and df.loc[sample-1, 'leftaxlebox'] > THRESHOLD:
        d = df.loc[sample-1,list(leftaxlebox_faults)].to_dict()
        highest_conf_fault = max(d.items(), key=operator.itemgetter(1))[0]
        leftaxlebox_string = f'_{highest_conf_fault}'
        print(sample, highest_conf_fault)

    rightaxlebox_string = '_RA0'
    if df.loc[sample-1, "rightaxlebox"] > THRESHOLD:
        rightaxlebox_string = f'_RA1'
  
    # REMOVE G4 if there are G2 and G8
    if '+G4' in gearbox_string and 'G2' in gearbox_string:
        gearbox_string = gearbox_string.replace('+G4','')
    if 'G4+' in gearbox_string and 'G8' in gearbox_string:
        gearbox_string = gearbox_string.replace('G4+','')

    submission_df.loc[sample-1, "Sample Number"] = sample
    submission_df.loc[sample-1, "motor"] = motor_string
    submission_df.loc[sample-1, "gearbox"] = gearbox_string
    submission_df.loc[sample-1, "leftaxlebox"] = leftaxlebox_string
    submission_df.loc[sample-1, "rightaxlebox"] = rightaxlebox_string

submission_df["Label"] = submission_df["motor"].astype(str) + submission_df["gearbox"].astype(str) + submission_df["leftaxlebox"].astype(str) + submission_df["rightaxlebox"].astype(str)

submission_df[["Sample Number", "Label"]].to_csv("Result_PDC0103.csv", index=False)
print(submission_df)
print(submission_df["Label"].value_counts())
