import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(csv_file):
    # 读取CSV文件
    data = pd.read_csv(csv_file)

    # 计算每一列之间的相关性
    correlation_matrix = data.corr()

    # 使用Seaborn绘制热力图
    sns.set(style="white")
    plt.figure(figsize=(10, 8))

    # 生成热力图
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=.5)

    # 显示图形
    plt.title('Correlation Heatmap')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot correlation heatmap for columns in a CSV file.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file.")
    
    args = parser.parse_args()

    # 调用绘图函数
    plot_correlation_heatmap(args.csv_file)
