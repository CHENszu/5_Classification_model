import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
def run_xgboost(X_train, y_train, X_predict):
    # 划分验证集
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # 配置模型参数
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train_split, y_train_split)

    # 验证集准确率
    val_accuracy = accuracy_score(y_val, model.predict(X_val))

    # 预测概率
    if len(X_predict) > 0:
        predict_proba = model.predict_proba(X_predict)
    else:
        predict_proba = None

    return {
        'name': 'XGBoost',
        'proba': predict_proba,
        'val_accuracy': val_accuracy
    }