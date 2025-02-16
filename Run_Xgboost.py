import pandas as pd
import xgboost as xgb
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

def main():
    # 获取文件路径
    file_path = input("请输入数据文件路径（xlsx/csv）: ").strip()

    # 读取数据
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("不支持的文件格式，请使用xlsx或csv")

    # 分割特征和标签
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 分割训练集和预测集
    train_mask = y.notna()
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_predict = X[~train_mask]

    if len(X_train) == 0:
        raise ValueError("数据中没有可用的训练样本（y列全为空）")

    # 标签编码和类别检测
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    num_classes = len(le.classes_)
    print(f"\n检测到分类类别数: {num_classes}")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_encoded, test_size=0.2, random_state=42
    )

    # 配置模型参数
    model_params = {
        'objective': 'multi:softmax' if num_classes > 2 else 'binary:logistic',
        'eval_metric': 'mlogloss' if num_classes > 2 else 'logloss',
        'num_class': num_classes if num_classes > 2 else None,
        'n_estimators': 100,
        'random_state': 42,
        'use_label_encoder': False
    }

    # 训练模型
    model = xgb.XGBClassifier(**model_params)
    model.fit(X_train, y_train)

    # 模型评估
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # 打印模型信息
    print("\n=== 模型详情 ===")
    print("模型参数:", model.get_params())
    print("\n混淆矩阵:")
    print(pd.DataFrame(conf_matrix,
                       index=le.classes_,
                       columns=le.classes_))
    print(f"\n测试集准确率: {accuracy:.4f}")

    # 特征重要性
    feature_importance = pd.Series(model.feature_importances_,
                                   index=X.columns).sort_values(ascending=False)
    print("\n特征重要性:")
    print(feature_importance)

    # 全量训练预测模型
    final_model = xgb.XGBClassifier(**model_params)
    final_model.fit(X[train_mask], y_encoded)

    # 进行预测
    if not X_predict.empty:
        predictions = le.inverse_transform(final_model.predict(X_predict))

        # 保存结果
        output_path = os.path.join(os.path.dirname(file_path), 'res_xgboost.xlsx')
        # 提取需要预测的完整数据行，并复制
        predict_data = df.loc[~train_mask].copy()
        # 替换因变量列为空值的部分为预测结果
        predict_data[y.name] = predictions
        # 保存到Excel文件，不保存索引
        predict_data.to_excel(output_path, index=False)
        print(f"\n预测结果已保存至: {output_path}")
    else:
        print("\n没有需要预测的数据")


if __name__ == "__main__":
    main()