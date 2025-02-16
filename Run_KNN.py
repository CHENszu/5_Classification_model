import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 提示用户输入文件路径
file_path = input("请输入文件路径（.xlsx或.csv）：")
file_extension = file_path.split('.')[-1].lower()

# 读取数据
if file_extension == 'csv':
    data = pd.read_csv(file_path)
elif file_extension in ['xls', 'xlsx']:
    data = pd.read_excel(file_path)
else:
    raise ValueError("不支持的文件格式，请输入.xlsx或.csv文件！")

# 获取列名
columns = data.columns.tolist()
# 最后一列是因变量y，前面的是自变量X
X = data[columns[:-1]]
y = data[columns[-1]]

# 检查是否有缺失值
if y.isnull().any():
    # 分离有标签和无标签的数据
    has_y = data[pd.notnull(y)].copy()
    no_y = data[pd.isnull(y)].copy()
else:
    raise ValueError("数据的因变量列中没有缺失值，请检查输入数据！")

# 分离有标签的数据用于训练和验证
X_train = has_y[columns[:-1]]
y_train = has_y[columns[-1]]

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(no_y[columns[:-1]])

# 训练KNN模型
knn = KNeighborsClassifier(n_neighbors=3)  # 可以调整k值
knn.fit(X_train_scaled, y_train)

# 预测
y_pred = knn.predict(X_test_scaled)

# 保存预测结果
no_y['预测值'] = y_pred
result = no_y.copy()

# 输出混淆矩阵和准确率等信息
print("KNN模型的训练结果：")
print("混淆矩阵：")
print(confusion_matrix(y_train, knn.predict(X_train_scaled)))
print("\n分类报告：")
print(classification_report(y_train, knn.predict(X_train_scaled)))
print(f"\n训练集准确率：{accuracy_score(y_train, knn.predict(X_train_scaled)):.4f}")

# 保存结果到 res_KNN.xlsx 文件
result_file_path = 'res_KNN.xlsx'
result.to_excel(result_file_path, index=False)
print(f"预测结果已保存到：{result_file_path}")