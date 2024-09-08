import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('output_gear_1000_real.csv')

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
