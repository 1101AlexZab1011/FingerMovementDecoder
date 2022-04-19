import pandas as pd
import numpy as np
import os

data1 = np.random.rand(10, 10)
data2 = np.random.rand(10, 10)
data3 = np.random.rand(10, 10)


root = '../'
# for data, sheet in zip([data1, data2, data3], ['sheet1', 'sheet2', 'sheet3']):
#     df = pd.DataFrame(data, columns=[f'col {i}' for i in range(data1.shape[0])])
#     with pd.ExcelWriter(os.path.join(root, 'testfile.xlsx')) as writer:
#         df.to_excel(writer, sheet_name=sheet, engine='openpyxl')
with pd.ExcelWriter(os.path.join(root, 'testfile.xlsx')) as writer:
    for data, sheet in zip([data1, data2, data3], ['sheet1', 'sheet2', 'sheet3']):
        df = pd.read_excel(os.path.join(root, 'testfile.xlsx'), sheet_name=sheet)
        print(df)
        # df = pd.DataFrame(data, columns=[f'col {i}' for i in range(data1.shape[0])])
        df.to_excel(writer, sheet_name=sheet, engine='openpyxl')