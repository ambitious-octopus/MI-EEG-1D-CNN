import pandas as pd

path = "D:\\TASK1.xlsx"
xls = pd.read_excel(path, sheet_name=None)
print(xls.keys())
