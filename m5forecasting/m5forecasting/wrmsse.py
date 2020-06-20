import pandas as pd
import numpy as np
from .loadData import *

def externalWeightCalc(vals):
    ## numpy array input
    arr = vals[:, -28:]
    total_sum = np.sum(arr)
    return np.sum(arr, axis = 1) / total_sum

def computeScale(data):
    idx = 0
    while (data[idx] == 0 and idx < len(data)):
        idx += 1
    num_terms = len(data) - idx - 1
    return np.sum(np.square(data[idx+1:] - data[idx:-1])) * 28.0 / num_terms

def computeAllScales(data):
    return np.sqrt(np.array(list(map(computeScale, data))))

def weightCalculator(demand_data, revenue_data):
    return externalWeightCalc(revenue_data) / computeAllScales(demand_data)

def runningSum(array, start, increment, startStep, stopStep):
    total = np.zeros(len(array[0]))
    for i in range(startStep, stopStep + 1):
        total += array[start + increment * i]
    return total

def Transform_12(vals):
    return vals

def Transform_11(vals):
    arr = np.zeros((9147, vals.shape[1]))
    for i in range(3049):
        arr[i] = runningSum(vals, i, 3049, 0, 3)
        arr[i + 3049] = runningSum(vals, i, 3049, 4, 6)
        arr[i + 3049 * 2] = runningSum(vals, i, 3049, 7, 9)
    return arr

def Transform_10(vals):
    arr = np.zeros((3049, vals.shape[1]))
    for i in range(3049):
        arr[i] = np.sum(vals[range(i, 30490, 3049)], axis = 0)
    return arr

current_id = ""
department_indices = []
for i in range(3049):
    if df.iloc[i]["dept_id"] != current_id:
        department_indices.append(i)
        current_id = df.iloc[i]["dept_id"]
department_indices.append(3049)

def Transform_9(vals):
    arr = np.zeros((70, vals.shape[1]))
    for i in range(70):
        arr[i] = np.sum(vals[range(3049 * (i//7), 3049 * ((i//7) + 1))][range(department_indices[i % 7], department_indices[i % 7 + 1])], axis = 0)
    return arr

current_id = ""
category_indices = []
for i in range(3049):
    if df.iloc[i]["cat_id"] != current_id:
        category_indices.append(i)
        current_id = df.iloc[i]["cat_id"]
category_indices.append(3049)

def Transform_8(vals):
    arr = np.zeros((30, vals.shape[1]))
    for i in range(30):
        arr[i] = np.sum(vals[range(3049 * (i//3), 3049 * ((i//3) + 1))][range(category_indices[i % 3], category_indices[i % 3 + 1])], axis = 0)
    return arr

def Transform_7(vals):
    new_vals = Transform_9(vals)
    arr = np.zeros((21, vals.shape[1]))
    for i in range(7):
        arr[i] = np.sum(new_vals[range(i, i + 7 * 4, 7)], axis = 0)
        arr[i + 7] = np.sum(new_vals[range(i + 7 * 4, i + 7 * 7, 7)], axis = 0)
        arr[i + 14] = np.sum(new_vals[range(i + 7 * 7, i + 7 * 10, 7)], axis = 0)
    return arr

def Transform_6(vals):
    new_vals = Transform_8(vals)
    arr = np.zeros((9, vals.shape[1]))
    for i in range(3):
        arr[i] = np.sum(new_vals[range(i, i + 3 * 4, 3)], axis = 0)
        arr[i + 3] = np.sum(new_vals[range(i + 3 * 4, i + 3 * 7, 3)], axis = 0)
        arr[i + 6] = np.sum(new_vals[range(i + 3 * 7, i + 3 * 10, 3)], axis = 0)
    return arr

def Transform_5(vals):
    arr = np.zeros((7, vals.shape[1]))
    for i in range(10):
        for j in range(7):
            arr[j] += np.sum(vals[3049 * i: 3049 * (i + 1)][department_indices[j]:department_indices[j+1]], axis = 0)
    return arr

def Transform_4(vals):
    arr = np.zeros((3, vals.shape[1]))
    for i in range(10):
        arr[0] += np.sum(vals[3049 * i: 3049 * (i + 1)][:565],axis = 0)
        arr[1] += np.sum(vals[3049 * i: 3049 * (i + 1)][565:1612],axis = 0)
        arr[2] += np.sum(vals[3049 * i: 3049 * (i + 1)][1612:],axis = 0)
    return arr

def Transform_3(vals):
    arr = np.zeros((10, vals.shape[1]))
    for i in range(10):
        arr[i] = np.sum(vals[3049 * i : 3049 * (i+1)], axis = 0)
    return arr

def Transform_2(vals):
    arr = np.zeros((3, vals.shape[1]))
    arr[0] = np.sum(vals[:12196], axis = 0)
    arr[1] = np.sum(vals[12196:21343], axis = 0)
    arr[2] = np.sum(vals[21343:], axis = 0)
    return arr

def Transform_1(vals):
    arr = np.zeros((1, vals.shape[1]))
    arr[0] = np.sum(vals, axis = 0)
    return arr

transformer = {}
for i in range(1,13):
    transformer[i] = globals()["Transform_{}".format(i)]

weights = {}
for i in range(1,13):
    weights[i] = weightCalculator(transformer[i](np.array(bare_df)), transformer[i](np.array(bare_revenue)))

class WRMSSE():
    def __init__(self, preds, actuals = np.array(actuals), demand = np.array(bare_df), revenue = np.array(bare_revenue), recomputeWeights = False):

        if isinstance(preds, pd.DataFrame):
            self.preds = clean(preds)
        else:
            self.preds = preds

        self.actuals = actuals
        self.demand = demand
        self.revenue = revenue
        self.weights = {}
        self.losses = {}
        if recomputeWeights:
            self.computeWeights()
        else:
            self.weights = weights

    def computeWeights(self):
        for i in range(1,13):
            self.weights[i] = weightCalculator(transformer[i](self.demand), transformer[i](self.revenue))

    def computeLevelLosses(self, level):
        if level not in self.losses.keys():
            self.losses[level] = self.WRMSSE(transformer[level](self.actuals), transformer[level](self.preds), self.weights[level])
        return self.losses[level]

    def computeLevelLoss(self, level):
        return self.computeLevelLosses(level).sum()

    def getLossByLevel(self):
        return np.array([self.computeLevelLoss(level) for level in range(1,13)])

    def getTotalLoss(self):
        total = 0
        for i in range(1,13):
            total += self.computeLevelLoss(i)
        return total

    def WRMSSE(self, actual, preds, weights):
        diff = actual - preds
        diff_squared = diff * diff
        return np.sqrt(np.sum(diff_squared, axis = 1)) * weights / 12.0
