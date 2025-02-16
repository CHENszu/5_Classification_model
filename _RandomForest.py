from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
def run_randomforest(X_train, y_train, X_predict):
    # 划分验证集
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # 训练模型
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_split, y_train_split)

    # 验证集准确率
    val_accuracy = accuracy_score(y_val, rf.predict(X_val))

    # 预测概率
    if len(X_predict) > 0:
        predict_proba = rf.predict_proba(X_predict)
    else:
        predict_proba = None

    return {
        'name': 'RandomForest',
        'proba': predict_proba,
        'val_accuracy': val_accuracy
    }