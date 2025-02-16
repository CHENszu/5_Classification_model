import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# 提示用户输入文件路径
file_path = input("请输入xlsx或csv文件的路径：")

# 读取数据
if file_path.endswith('.xlsx'):
    data = pd.read_excel(file_path)
elif file_path.endswith('.csv'):
    data = pd.read_csv(file_path)
else:
    print("不支持的文件格式，请输入xlsx或csv文件。")
    exit()

# 分离自变量X和因变量y
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 找到需要预测的数据（最后一部分因变量为空的数据）
predict_index = y[y.isnull()].index
X_predict = X.loc[predict_index]

# 去除需要预测的数据，得到训练数据
train_index = y[~y.isnull()].index
X_train = X.loc[train_index]
y_train = y.loc[train_index]

# 创建随机森林分类器
rf = RandomForestClassifier(random_state=42)

# 训练模型
rf.fit(X_train, y_train)

# 打印模型信息
print("拟合参数：")
print(rf.get_params())

# 划分训练集和测试集进行评估
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
rf_split = RandomForestClassifier(random_state=42)
rf_split.fit(X_train_split, y_train_split)
y_pred = rf_split.predict(X_test_split)

# 混淆矩阵
print("\n混淆矩阵：")
print(confusion_matrix(y_test_split, y_pred))

# 模型准确率
accuracy = accuracy_score(y_test_split, y_pred)
print(f"\n模型准确率：{accuracy:.2f}")

# 各列数据的权重
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index=X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print("\n各列数据的权重：")
print(feature_importances)

# 打印分类报告
print("\n分类报告：")
print(classification_report(y_test_split, y_pred))

# 进行预测
predictions = rf.predict(X_predict)

# 创建结果 DataFrame，包括预测的因变量和预测结果
# 保留原始数据中的其他列和需要预测的因变量列
result_df = data.loc[predict_index].copy()  # 保留原始数据的其他列和需要预测的因变量列
result_df[data.columns[-1]] = predictions   # 替换因变量列为空值的部分为预测结果

# 保存分类结果
save_path = os.path.join(os.path.dirname(file_path), 'res_RandomForest.xlsx')
result_df.to_excel(save_path, index=False)  # 保存结果，包括其他列和因变量列

print(f"\n分类结果已保存到：{save_path}")