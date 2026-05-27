import pandas as pd
import numpy as np
from scipy.integrate import simpson
import os
import json

low_limit = 0.001
high_limit = 0.99

def compute_adsorption_ratio(file_path):
    file_1 = os.path.join(file_path, 'component_1_Compent1.data')
    file_2 = os.path.join(file_path, 'component_2_Compent2.data')
    params_file = os.path.join(file_path, 'params.json')

    result1 = pd.read_csv(file_1, sep=' ')
    result2 = pd.read_csv(file_2, sep=' ')

    with open(params_file, 'r') as pf:
        params = json.load(pf)

    valid_result1 = result1[result1.iloc[:, 2] > low_limit]
    valid_result2 = result2[result2.iloc[:, 2] > low_limit]

    if valid_result1.empty or valid_result2.empty:
        raise ValueError("筛选后数据为空")

    t1 = valid_result1.iloc[0, 0]
    t2 = valid_result2.iloc[0, 0]

    if t1 > t2:
        t1, t2 = t2, t1

    if t1 == t2:
        raise ValueError("t1 和 t2 相等")

    filtered_result1 = result1[(result1.iloc[:, 1] > t1) & (result1.iloc[:, 1] < t2)]

    if filtered_result1.empty:
        raise ValueError("t1 到 t2 之间无有效数据")

    x_values_1 = filtered_result1.iloc[:, 1]
    y_values_1 = filtered_result1.iloc[:, 2]

    integral_1 = simpson(y=y_values_1, x=x_values_1)

    component_1_adsorption_ratio = t2 - integral_1

    return component_1_adsorption_ratio, t2, params



def compute_on_filtered_folders(root_dir):
    filtered_folder_name = pd.read_csv('filtered_result_new.csv')
    dataset = []
    dataset_target = []

    success_count = 0
    fail_count = 0

    for folder_name in filtered_folder_name.iloc[:, 1]:
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            try:
                res1, res2, params = compute_adsorption_ratio(folder_path)
                dataset.extend(params)
                dataset_target.append([res1, res2])
                success_count += 1
            except Exception as e:
                # print(f"[跳过] 文件夹：{folder_name} 处理失败，原因：{e}")
                fail_count += 1
    
    print(f"成功处理：{success_count} 个文件夹，失败：{fail_count} 个文件夹")

    np_dataset = np.array(dataset).reshape(-1, 24)
    df_dataset_target = pd.DataFrame(dataset_target, columns=['absorption_ratio', 'time'])

    np.savetxt('time_dataset.csv', np_dataset, delimiter=',', fmt='%f')
    df_dataset_target.to_csv('time.csv', index=False)


root_directory = '/home/ls/桌面/RUPTURA-main/work/break_new'
compute_on_filtered_folders(root_directory)
