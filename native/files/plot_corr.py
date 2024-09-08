import argparse
import pandas as pd

def calculate_correlation(csv_file):
    # 读取CSV文件
    data = pd.read_csv(csv_file)

    # 计算每一列之间的相关性
    correlation_matrix = data.corr()

    # 输出相关性矩阵
    print("Correlation matrix:")
    print(correlation_matrix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate column correlations in a CSV file.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file.")
    
    args = parser.parse_args()

    # 调用计算相关性的函数
    calculate_correlation(args.csv_file)
