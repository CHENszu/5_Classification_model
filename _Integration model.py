import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from _Logistic_Lasso import run_logistic_lasso
from _KNN import run_knn
from _RandomForest import run_randomforest
from _Xgboost import run_xgboost
from _BP import run_bp
# 模型映射表
MODEL_MAP = {
    '1': ('Logistic_Lasso', run_logistic_lasso),
    '2': ('KNN', run_knn),
    '3': ('RandomForest', run_randomforest),
    '4': ('XGBoost', run_xgboost),
    '5': ('BP_Network', run_bp)
}

def main():
    # 1. 读取数据
    file_path = input("请输入数据文件路径（xlsx/csv）: ").strip()
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("不支持的文件格式！")

    # 2. 预处理
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values
    le = LabelEncoder()
    mask = ~pd.isnull(y)
    y[mask] = le.fit_transform(y[mask].astype(str))
    y = pd.to_numeric(y, errors='coerce')

    # 3. 分割预测数据
    predict_mask = pd.isnull(y)
    X_train = X[mask]
    y_train = y[mask].astype(int)
    X_predict = X[predict_mask]

    # 4. 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_predict_scaled = scaler.transform(X_predict) if len(X_predict) > 0 else None

    # 5. 用户选择模型
    print('1-Run_Logistic_Lasso\n2-KNN\n3-Run_RandomForest\n4-Xgboost\n5-BP network')
    model_choice = input("请选择模型（1-5，逗号分隔，如1,3,5）: ").strip().split(',')
    selected_models = [MODEL_MAP[num] for num in model_choice if num in MODEL_MAP]

    # 6. 运行模型并收集结果
    results = []
    for model_name, model_func in selected_models:
        print(f"\n正在运行模型: {model_name}...")
        result = model_func(X_train_scaled, y_train, X_predict_scaled)
        results.append(result)
        print(f"{model_name}验证集准确率: {result['val_accuracy']:.4f}")

    # 7. 计算加权平均概率
    weights = np.array([res['val_accuracy'] for res in results])
    weights /= weights.sum()  # 归一化权重

    final_proba = np.zeros((len(X_predict), len(le.classes_)))
    for res, weight in zip(results, weights):
        if res['proba'] is not None:
            final_proba += res['proba'] * weight

    # 8. 保存概率到Excel的不同sheet
    # with pd.ExcelWriter('res_p.xlsx') as writer:
    #     for res in results:
    #         if res['proba'] is not None:
    #             df_proba = pd.DataFrame(
    #                 res['proba'],
    #                 columns=[f"{res['name']}_Class_{i}" for i in le.classes_]
    #             )
    #             df_proba.to_excel(writer, sheet_name=res['name'], index=False)
    # 8. 保存概率到Excel的不同sheet，并添加加权平均结果
    with pd.ExcelWriter('res_p.xlsx') as writer:
        # 保存各模型概率
        for res in results:
            if res['proba'] is not None:
                df_proba = pd.DataFrame(
                    res['proba'],
                    columns=[f"{res['name']}_Class_{i}" for i in le.classes_]
                )
                df_proba.to_excel(writer, sheet_name=res['name'], index=False)

        # 新增加权平均概率sheet
        if len(X_predict) > 0 and final_proba.shape[0] > 0:
            df_final = pd.DataFrame(
                final_proba,
                columns=[f"Weighted_Avg_Class_{i}" for i in le.classes_]
            )
            df_final.to_excel(writer, sheet_name="Weighted_Average", index=False)
    print("\n详细概率信息已保存至 res_p.xlsx")
    # 9. 保存最终预测结果
    final_labels = le.inverse_transform(np.argmax(final_proba, axis=1))
    predict_df = df.loc[predict_mask].copy()
    predict_df[df.columns[-1]] = final_labels
    predict_df.to_excel('res.xlsx', index=False)
    print("\n预测结果已保存至 res.xlsx")

if __name__ == "__main__":
    main()