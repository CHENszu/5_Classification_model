from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore')

def run_logistic_lasso(X_train, y_train, X_predict):
    # 划分验证集
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42
    )

    # 训练模型
    model = LogisticRegression(
        penalty='l1',
        solver='saga',  # saga支持多分类和L1正则化
        multi_class='auto',
        max_iter=10000,
        random_state=42
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_train_split, y_train_split)

    # 输出模型信息
    print("\n========== 模型摘要 ==========")
    print(f"迭代次数: {model.n_iter_}")
    print(f"收敛状态: {'成功' if model.n_iter_[0] < model.max_iter else '未完全收敛'}")

    # 验证集准确率
    val_accuracy = accuracy_score(y_val, model.predict(X_val))

    # 预测概率
    if len(X_predict) > 0:
        predict_proba = model.predict_proba(X_predict)
    else:
        predict_proba = None

    return {
        'name': 'Logistic_Lasso',
        'proba': predict_proba,
        'val_accuracy': val_accuracy
    }
