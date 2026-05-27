import matplotlib.pyplot as plt
import json
import pandas as pd
import os

def isotherms(i):
    try:
        # 打开JSON文件
        file = open(f'./break/{i}/params.json', 'r', encoding='utf-8')

        # 读取数据
        data = json.load(file)

        # 打印数据
        print(data)

        # 关闭文件
        file.close()
        pressure = [1.0, 11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 71.0, 81.0, 91.0, 101.0]
        a11 = data[9]
        b11 = data[10]
        c11 = data[11]
        a12 = data[12]
        b12 = data[13]
        c12 = data[14]
        a21 = data[18]
        b21 = data[19]
        c21 = data[20]
        a22 = data[21]
        b22 = data[22]
        c22 = data[23]
        iso1 = [(a11*(b11*x/((1+((b11*x)**c11))**(1/c11)))) + (a12*(b12*x/((1+((b12*x)**c12))**(1/c12)))) for x in pressure]
        iso2 = [(a21*(b21*x/((1+((b21*x)**c21))**(1/c21)))) + (a22*(b22*x/((1+((b22*x)**c22))**(1/c22)))) for x in pressure]
        # 创建图形和轴对象
        plt.figure()
        plt.plot(pressure, iso1, label='isotherm1', marker='o')
        plt.plot(pressure, iso2, label='isotherm2', marker='s', linestyle='--')

        # 添加标题和标签
        plt.title('isotherm')
        plt.xlabel('pressure')
        plt.ylabel('ammount')

        # 添加图例
        plt.legend()
        plt.savefig(f'./breakdata/{i}_isotherm.jpg')
    except:
        pass

def breakthrough(i):
    try:
        # 使用pandas的read_csv函数读取数据文件
        # sep参数指定了列之间的分隔符，这里是空格
        # header=None表示文件中没有列标题行
        data1 = pd.read_csv(f'./break/{i}/component_1_Compent1.data', sep=' ', header=None)
        data2 = pd.read_csv(f'./break/{i}/component_2_Compent2.data', sep=' ', header=None)
        # 创建图形和轴对象
        plt.figure()
        plt.plot(data1.iloc[:, 0], data1.iloc[:, 2], label='isotherm1', marker='o')
        plt.plot(data2.iloc[:, 0], data2.iloc[:, 2], label='isotherm2', marker='s', linestyle='--')

        # 添加标题和标签
        plt.title('Breakthrough')
        plt.xlabel('Time')
        plt.ylabel('c/c0')

        # 添加图例
        plt.legend()
        plt.savefig(f'./breakdata/{i}_breakthrough.jpg')
    except:
        pass

for i in os.listdir("./break/"):
    isotherms(i)
    breakthrough(i)