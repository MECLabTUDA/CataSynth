from sklearn.metrics import matthews_corrcoef
import pandas as pd
import numpy as np

xls = pd.ExcelFile("User_study_gt.xlsx")

df_GT = pd.read_excel(xls, 'Sheet2')
predictions = []
for file in [f"User_study_{id}.xlsx" for id in ["TD", "FU", "MF", "JP", "AG", "FW"]]:
    xls = pd.ExcelFile(file)
    predictions.append(pd.read_excel(xls, 'Sheet2').fillna(0))

coefficients = []
false_class = []
for name, pred in zip(["TD", "FU", "MF", "JP", "AG", "FW"], predictions):
    print("Matthews correlation coefficient (MCC) for", name, matthews_corrcoef(df_GT["L"], pred["L"]))
    coefficients.append(matthews_corrcoef(df_GT["L"], pred["L"]))
    print("False classifications for", name, sum(np.absolute(df_GT["L"] - pred["L"])), "/50")
    false_class.append(sum(np.absolute(df_GT["L"] - pred["L"])))

print("Avg. MCC: ", np.mean(coefficients))
print(f"Avg. false classifications: {np.mean(false_class)/50}")
