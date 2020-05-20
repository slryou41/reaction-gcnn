import pickle
import matplotlib.pyplot as plt
import numpy as np

import xlrd
import csv
import pandas as pd
import json


# TODO
# 1. Final test ids (After filtering out RDKit non-readable reactions)
# 2. Generate a set of predictions and ground truths
# 3. Visualize the performances

# Data format:
# [{id: int,
#   pred: {'type1': [top1, top2, top3, top4, top5], 'type2': .. },
#   gt: {'type1': [], 'type2': [], ..., 'other': [] }, {}, ..
# ]
# Note that null class should be in gt when there's no ground truth reagent.
# Null class should be assigned as an integer id
# type_count needed! (for binary classification)
# type_count = {'type1': # of labels (including null), 'type2': # of labels, .. }

def compute_PR(data, type_count, K=3):
    
    type_names = type_count.keys()
    
    gt_per_class = dict()    # # of ground truth samples in each class
    
    corr_per_class = dict()  # # of correctly predicted samples in each class
    pred_per_class = dict()  # # of predicted samples in each class
    corr_per_class_topk = dict()  
    pred_per_class_topk = dict() 
    
    for ii, r_name in enumerate(type_names):
        if type_count[r_name] != 1:
            gt_per_class.update({r_name+str(x+1): 0 for x in range(type_count[r_name])})
            corr_per_class.update({r_name+str(x+1): 0 for x in range(type_count[r_name])})
            pred_per_class.update({r_name+str(x+1): 0 for x in range(type_count[r_name])})
            corr_per_class_topk.update({r_name+str(x+1): 0 for x in range(type_count[r_name])})
            pred_per_class_topk.update({r_name+str(x+1): 0 for x in range(type_count[r_name])})
        else:
            gt_per_class.update({r_name: 0 for x in range(type_count[r_name])})
            corr_per_class.update({r_name: 0 for x in range(type_count[r_name])})
            pred_per_class.update({r_name: 0 for x in range(type_count[r_name])})
            corr_per_class_topk.update({r_name: 0 for x in range(type_count[r_name])})
            pred_per_class_topk.update({r_name: 0 for x in range(type_count[r_name])})

        
    for i in range(len(data)):
        for r_type in type_names:
            
            r_gt = set(data[i]['gt'][r_type])
            r_pred = data[i]['pred'][r_type]

            if type_count[r_type] > 1:
                for g in r_gt:
                    gt_per_class[r_type+str(g)] += 1
                # top-k
                corr_label = [v for v in r_pred[:K] if v in r_gt]

                for p in r_pred[:K]:
                    pred_per_class_topk[r_type+str(p)] += 1
                for c in corr_label:
                    corr_per_class_topk[r_type+str(c)] += 1

                # top-1
                pred_per_class[r_type+str(r_pred[0])] += 1
                if r_pred[0] in r_gt:
                    corr_per_class[r_type+str(r_pred[0])] += 1
                    
            else: # binary classification
                gt_per_class[r_type] += 1
                
                corr_label = [v for v in r_pred if v in r_gt]
                
                for p in r_pred:
                    pred_per_class_topk[r_type] += 1
                    pred_per_class[r_type] += 1
                
                if len(corr_label) > 0:
                    corr_per_class[r_type] += 1
                    corr_per_class_topk[r_type] += 1
                
    PCP = {x: float(corr_per_class[x])/max(1, pred_per_class[x]) for x in pred_per_class.keys()} # per-class precision
    PCR = {x: float(corr_per_class[x])/max(1, gt_per_class[x]) for x in gt_per_class.keys()} # per-class recall
    PCP_topk = {x: float(corr_per_class_topk[x])/max(1, pred_per_class_topk[x]) for x in pred_per_class_topk.keys()}
    PCR_topk = {x: float(corr_per_class_topk[x])/max(1, gt_per_class[x]) for x in gt_per_class.keys()}
    
    AP = dict()
    AR = dict()
    OVP = dict() # overall precision
    OVR = dict() # overall recall
    
    AP_topk = dict()
    AR_topk = dict()
    OVP_topk = dict()
    OVR_topk = dict()
    
    for ii, r_name in enumerate(type_names):
        if type_count[r_name] > 1:
            r_keys = [r_name+str(x+1) for x in range(type_count[r_name])]
        else:
            r_keys = [r_name]
        
        precisions = [PCP[k] for k in r_keys]
        AP[r_name] = sum(precisions) / float(type_count[r_name])
        recalls = [PCR[k] for k in r_keys]
        AR[r_name] = sum(recalls) / float(type_count[r_name])
        
        r_corr_num = [corr_per_class[x] for x in r_keys]
        r_pred_num = [pred_per_class[x] for x in r_keys]
        r_gt_num = [gt_per_class[x] for x in r_keys]
        
        OVP[r_name] = sum(r_corr_num) / float(max(1, sum(r_pred_num)))
        OVR[r_name] = sum(r_corr_num) / float(max(1, sum(r_gt_num)))
        
        precisions = [PCP_topk[k] for k in r_keys]
        AP_topk[r_name] = sum(precisions) / float(type_count[r_name])
        recalls = [PCR_topk[k] for k in r_keys]
        AR_topk[r_name] = sum(recalls) / float(type_count[r_name])
        
        r_corr_num = [corr_per_class_topk[x] for x in r_keys]
        r_pred_num = [pred_per_class_topk[x] for x in r_keys]
        r_gt_num = [gt_per_class[x] for x in r_keys]
        
        OVP_topk[r_name] = sum(r_corr_num) / float(max(1, sum(r_pred_num)))
        OVR_topk[r_name] = sum(r_corr_num) / float(max(1, sum(r_gt_num)))
        
    AP['all'] = sum(PCP.values()) / float(sum(type_count.values()))
    AR['all'] = sum(PCR.values()) / float(sum(type_count.values()))
    OVP['all'] = sum(corr_per_class.values()) / float(sum(pred_per_class.values()))
    OVR['all'] = sum(corr_per_class.values()) / float(sum(gt_per_class.values()))
    
    AP_topk['all'] = sum(PCP_topk.values()) / float(sum(type_count.values()))
    AR_topk['all'] = sum(PCR_topk.values()) / float(sum(type_count.values()))
    OVP_topk['all'] = sum(corr_per_class_topk.values()) / float(sum(pred_per_class_topk.values()))
    OVR_topk['all'] = sum(corr_per_class_topk.values()) / float(sum(gt_per_class.values()))
    
    return (PCP, PCR, AP, AR, OVP, OVR), (PCP_topk, PCR_topk, AP_topk, AR_topk, OVP_topk, OVR_topk) 


# Per-sample accuracy in each category
def accuracy(data, type_count, K=3):
    
    total_num = {x: 0 for x in type_count.keys()}
    top1 = {x: 0 for x in type_count.keys()}
    topk = {x: 0 for x in type_count.keys()}
    
    type_names = type_count.keys()
    
    for i in range(len(data)):
        for r_type in type_names:
            r_gt = data[i]['gt'][r_type]
            r_pred = data[i]['pred'][r_type]
            
            if type_count[r_type] > 1:
                # len(r_gt) should be always greater than 0!!
                assert len(r_gt) > 0
                
                total_num[r_type] += 1
                if r_pred[0] in r_gt:
                    top1[r_type] += 1
                
                len_intersect = len([v for v in r_pred[:K] if v in r_gt])
                if len_intersect > 0:
                    topk[r_type] += 1  # soft top-5
                    
            else:  # binary classification -> accuracy
                total_num[r_type] += 1
                    
                if len(r_pred) > 0 and len(r_gt) > 0:
                    top1[r_type] += 1
                if len(r_pred) == 0 and len(r_gt) == 0:
                    top1[r_type] += 1
                topk[r_type] += 1  # Always correct
    
    top1_acc = {x: float(top1[x])/total_num[x] for x in type_count.keys()}
    topk_acc = {x: float(topk[x])/total_num[x] for x in type_count.keys()}
    
    for t in type_names:
        print("Top 1 accuracy for type ", t, " :", top1_acc[t])
        print("Top k accuracy for type ", t, " :", topk_acc[t])
        
    return top1_acc, topk_acc

if __name__ == "__main__":
    
    data_file = json.load(open("data_file2.json", "r"))
    type_count = {'M': 45, 'L': 48, 'B': 14, 'S': 23, 'A': 75, 'other': 1}
    top1_acc, topk_acc = accuracy(data_file, type_count)

    result = compute_PR(data_file, type_count)
    print(result)