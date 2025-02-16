import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore')

def main():
    # 获取文件路径
    file_path = input("请输入数据文件的路径（.xlsx 或 .csv）: ").strip()
    if not os.path.exists(file_path):
        print("文件不存在，请检查路径！")
        return

    # 读取数据
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        print("仅支持.xlsx和.csv文件格式！")
        return

    # 分割特征和标签
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 分离训练集和预测集
    mask = y.isna()
    X_train = X[~mask]
    y_train = y[~mask].astype(int)  # 转换为整数类型
    X_predict = X[mask]

    # 检查数据有效性
    if len(X_train) == 0:
        print("训练数据不足，请确保最后一列包含已标注数据！")
        return

    # 自动确定分类数量
    n_classes = len(y_train.unique())
    print(f"\n检测到数据包含 {n_classes} 个类别: {sorted(y_train.unique())}")

    # 配置模型参数
    model = LogisticRegression(
        penalty='l1',
        solver='saga',  # saga支持多分类和L1正则化
        multi_class='auto',
        max_iter=10000,
        random_state=42
    )

    # 训练模型（抑制收敛警告）
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_train, y_train)

    # 训练集预测结果
    y_pred = model.predict(X_train)

    # 输出模型信息
    print("\n========== 模型摘要 ==========")
    print(f"迭代次数: {model.n_iter_}")
    print(f"收敛状态: {'成功' if model.n_iter_[0] < model.max_iter else '未完全收敛'}")

    # 输出分类指标
    print("\n========== 分类指标 ==========")
    print("混淆矩阵:")
    print(confusion_matrix(y_train, y_pred))
    print("\n准确率:", accuracy_score(y_train, y_pred))
    print("\n分类报告:")
    print(classification_report(y_train, y_pred))

    # 输出特征权重
    print("\n========== 特征权重 ==========")
    for i, class_label in enumerate(model.classes_):
        print(f"\n类别 {class_label} 的参数:")
        for feature, coef in zip(X.columns, model.coef_[i]):
            print(f"{feature}: {coef:.4f}")
        print(f"截距项: {model.intercept_[i]:.4f}")

    # 进行预测并保存结果
    if not X_predict.empty:
        y_predictions = model.predict(X_predict)

        # 创建一个新的 DataFrame，只包含预测的部分
        df_filtered = df[mask].copy()  # 只保留需要预测的行
        df_filtered[df.columns[-1]] = y_predictions.astype(int)  # 替换标签列

        # 保存结果文件，只保存预测的部分
        output_path = os.path.join(os.path.dirname(file_path), "res_Logistic.xlsx")
        df_filtered.to_excel(output_path, index=False)
        print(f"\n预测结果已保存至: {output_path}")
    else:
        print("\n没有需要预测的数据")


if __name__ == "__main__":
    main()