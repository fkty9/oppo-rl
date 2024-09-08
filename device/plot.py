import pandas as pd
import matplotlib.pyplot as plt

file_path = 'output_PPO_1000.csv'  # 替换为你的CSV文件路径

def plot_csv_columns(file_path):
    # 读取CSV文件
    data = pd.read_csv(file_path)
    
    # 获取列名
    columns = data.columns
    
    # 为每一列生成折线图
    for column in columns:
        plt.figure()
        plt.plot(data[column])
        plt.title(f"Line plot for {column}")
        plt.xlabel('Index')
        plt.ylabel(column)
        plt.grid(True)
        plt.savefig('pics/' + column + '.png')
        # plt.show()

# 示例使用
plot_csv_columns(file_path)

df = pd.read_csv(file_path)

# 检查行数是否足够
if len(df) < 300:
    print("数据行数不足300行")
else:
    # 计算每一列的后300行的平均值
    averages = df.iloc[-300:].mean()

    # 输出每个字段及其对应的平均值
    print("后300行的平均值:")
    for column, avg in averages.items():
        print(f"{column}: {avg}")