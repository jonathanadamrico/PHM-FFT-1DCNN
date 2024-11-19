#metrics.py

# Given a multi class label,
# Calculate the accuracy, precision, recall, f1, and z_metric

import numpy as np

type_to_fault = {
    "TYPE0":"M0_G0_LA0_RA0",
    "TYPE1":"M1_G0_LA0_RA0",
    "TYPE2":"M2_G0_LA0_RA0",
    "TYPE3":"M3_G0_LA0_RA0",
    "TYPE4":"M4_G0_LA0_RA0",
    "TYPE5":"M0_G1_LA0_RA0",
    "TYPE6":"M0_G2_LA0_RA0",
    "TYPE7":"M0_G3_LA0_RA0",
    "TYPE8":"M0_G4_LA0_RA0",
    "TYPE9":"M0_G5_LA0_RA0",
    "TYPE10":"M0_G6_LA0_RA0",
    "TYPE11":"M0_G7_LA0_RA0",
    "TYPE12":"M0_G8_LA0_RA0",
    "TYPE13":"M0_G0_LA1_RA0",
    "TYPE14":"M0_G0_LA2_RA0",
    "TYPE15":"M0_G0_LA3_RA0",
    "TYPE16":"M0_G0_LA4_RA0"
}

motor_labels = ("M0","M1","M2","M3","M4")
gearbox_labels = ("G0","G1","G2","G3","G4","G5","G6","G7","G8")
leftaxlebox_labels = ("LA0","LA1","LA2","LA3","LA4")
rightaxlebox_labels = ("RA0","RA1")

fault_labels = ("M0","M1","M2","M3","M4",
                "G0","G1","G2","G3","G4","G5","G6","G7","G8",
                "LA0","LA1","LA2","LA3","LA4",
                "RA0","RA1")

def convert_type_to_fault(fault_type):
    return type_to_fault[fault_type]

# The labels are by default in string of format 'M0_G0_LA0_RA0'
# Convert this into a list ['M0','G0','LA0','RA0']
def convert_label_to_list(label):
    new_label = label.replace('+','_')
    return new_label.split('_')

# This only calculates for one (1) sample
def calc_truefalse_posneg(true_label:str, pred_label:str, all_labels=fault_labels, component=None, test_set=False):
    if test_set:
        true_label = convert_type_to_fault(true_label)
    true_label = convert_label_to_list(true_label)
    pred_label = convert_label_to_list(pred_label)
    if component in ('M','G','LA','RA'):
        true_label = [x for x in true_label if x.startswith(component)]
        pred_label = [x for x in pred_label if x.startswith(component)]
    intersection = set(pred_label).intersection(set(true_label))
    union = set(pred_label).union(set(true_label))
    TP = intersection
    TN = set(all_labels).difference(union)
    FP = set(pred_label).difference(TP)
    FN = set(true_label).difference(TP)
    return TP, TN, FP, FN

def accuracy_score(TP_count, TN_count, FP_count, FN_count):
    return (TP_count+TN_count) / (TP_count+TN_count+FP_count+FN_count)

def precision_score(TP_count, FP_count):
    return TP_count / (TP_count + FP_count)

def recall_score(TP_count, FN_count):
    return TP_count / (TP_count + FN_count)

def f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

def z_metric(accuracy, precision, recall, f1):
    return 0.4*accuracy + 0.2*precision + 0.2*recall + 0.2*f1

def is_valid_label(pred_label):
    # if there is a component normal in predictio, there should be no anomaly in that component
    if "RA0" in pred_label and "RA1" in pred_label:
        print(pred_label)
        return False
    if "LA0" in pred_label and len(set([x for x in pred_label if x.startswith("LA")])) > 1:
        print(pred_label)
        return False
    if "G0" in pred_label and len(set([x for x in pred_label if x.startswith("G")])) > 1:
        print(pred_label)
        return False
    if "M0" in pred_label and len(set([x for x in pred_label if x.startswith("M")])) > 1:
        print(pred_label)
        return False
    return True

def calc_overall_metric(true_labels:iter, pred_labels:iter, all_labels=fault_labels, component=None, test_set=False):
    assert len(true_labels) == len(pred_labels)
    TP_count = 0
    TN_count = 0
    FP_count = 0
    FN_count = 0
    for i in range(len(true_labels)):
        assert is_valid_label(pred_labels[i])
        TP, TN, FP, FN = calc_truefalse_posneg(true_labels[i], pred_labels[i], all_labels, component, test_set)
        TP_count += len(TP)
        TN_count += len(TN)
        FP_count += len(FP)
        FN_count += len(FN)

    accuracy = accuracy_score(TP_count, TN_count, FP_count, FN_count)
    precision = precision_score(TP_count, FP_count)
    recall = recall_score(TP_count, FN_count)
    f1 = f1_score(precision, recall)
    return z_metric(accuracy, precision, recall, f1)

'''def calc_overall_metric(true_labels:iter, pred_labels:iter, all_labels=fault_labels, component=None, test_set=False):
    assert len(true_labels) == len(pred_labels)

    z_metrics = []
    for i in range(len(true_labels)):
        assert is_valid_label(pred_labels[i])
        TP, TN, FP, FN = calc_truefalse_posneg(true_labels[i], pred_labels[i], all_labels, component, test_set)
        TP_count = len(TP)
        TN_count = len(TN)
        FP_count = len(FP)
        FN_count = len(FN)

        accuracy = accuracy_score(TP_count, TN_count, FP_count, FN_count)
        precision = precision_score(TP_count, FP_count)
        recall = recall_score(TP_count, FN_count)
        f1 = f1_score(precision, recall)
        z_metrics.append(z_metric(accuracy, precision, recall, f1))

    return np.mean(z_metrics) '''