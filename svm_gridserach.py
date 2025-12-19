import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# 尝试导入进度条库，如果没有安装则使用简易版
try:
    from tqdm import tqdm
except ImportError:
    # 如果没装 tqdm，定义一个假的进度条函数，直接返回迭代器
    def tqdm(iterator, desc=""):
        return iterator
    print("提示: 安装 'tqdm' 库 (pip install tqdm) 可以看到漂亮的进度条")

# ========== 1. 核心工具函数 ==========

def img2vector(file_path):
    """
    将 32x32 的文本图片转换为 1x1024 的向量
    """
    return_vect = np.zeros((1, 1024))
    try:
        with open(file_path, 'r') as fr:
            for i in range(32):
                line_str = fr.readline().strip()
                # 容错处理：确保每行长度足够，不足补0
                if len(line_str) < 32:
                    line_str = line_str.ljust(32, '0')
                for j in range(32):
                    return_vect[0, 32 * i + j] = int(line_str[j])
    except Exception as e:
        print(f"读取文件错误 {file_path}: {e}")
    return return_vect

def load_dataset(dir_path, dataset_name="数据集"):
    """
    加载数据集并显示进度条
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"找不到路径: {dir_path}")

    file_list = [f for f in os.listdir(dir_path) if f.endswith('.txt')]
    m = len(file_list)
    
    # 初始化数据矩阵
    training_mat = np.zeros((m, 1024))
    hw_labels = []

    print(f"正在加载 {dataset_name} ({m} 个样本)...")
    # 使用 tqdm 显示进度条
    for i in tqdm(range(m), desc="读取文件"):
        file_name_str = file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        
        hw_labels.append(class_num_str)
        training_mat[i, :] = img2vector(os.path.join(dir_path, file_name_str))
        
    return training_mat, np.array(hw_labels)

# ========== 2. 可视化工具 ==========

def plot_confusion_matrix(model, X_test, y_test):
    """
    绘制混淆矩阵（加分项：展示模型在哪两个数字之间容易混淆）
    """
    print("\n正在生成混淆矩阵图...")
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay.from_estimator(
        model, 
        X_test, 
        y_test, 
        cmap=plt.cm.Blues, 
        ax=ax,
        values_format='d' # 显示整数
    )
    disp.ax_.set_title("SVM Recognition Confusion Matrix")
    plt.show()

def show_prediction_examples(X_test, y_test, y_pred, num_examples=5):
    """
    展示几个预测成功和失败的样本（直观展示）
    """
    indices = np.random.choice(len(y_test), num_examples, replace=False)
    
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        ax = plt.subplot(1, num_examples, i + 1)
        # 将 1x1024 还原回 32x32 图片
        img = X_test[idx].reshape(32, 32)
        
        true_label = y_test[idx]
        pred_label = y_pred[idx]
        
        # 预测错误标红，正确标绿
        color = 'green' if true_label == pred_label else 'red'
        
        ax.imshow(img, cmap='gray_r') # 黑白反转显示
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
        ax.axis('off')
    plt.suptitle("Random Prediction Examples")
    plt.show()

# ========== 3. 主程序流程 ==========

def main():
    # ************************** 路径设置 **************************
    # 请确保这里路径正确
    train_dir = r"c:\Users\E507\Documents\GitHub\svm\dataset\trainingDigits"    
    test_dir  = r"c:\Users\E507\Documents\GitHub\svm\dataset\testDigits"        
    # ************************************************************

    # 1. 加载数据
    try:
        X_train, y_train = load_dataset(train_dir, "训练集")
        X_test, y_test = load_dataset(test_dir, "测试集")
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("请检查代码中的 'train_dir' 和 'test_dir' 路径是否正确！")
        return

    # 2. 定义参数网格
    # rbf核对 gamma 和 C 非常敏感
    # C: 惩罚系数。C越大，越不能容忍错误（容易过拟合）；C越小，越容易欠拟合。
    # gamma: 决定了支持向量的影响范围。gamma越大，特征分布越窄（容易过拟合）。
    param_grid = [
        {'C': [1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    # 3. 网格搜索
    print("\n========== 开始模型训练与参数搜索 ==========")
    svc = SVC(random_state=42)
    clf = GridSearchCV(
        svc, 
        param_grid, 
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1,  # 使用所有CPU核心
        verbose=1
    )
    clf.fit(X_train, y_train)

    print("\n========== 搜索结果 ==========")
    print(f"最佳参数组合: {clf.best_params_}")
    print(f"交叉验证最高分: {clf.best_score_:.4f}")

    # 4. 最终评估
    best_model = clf.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n========== 测试集最终评估 ==========")
    print(f"测试集准确率: {acc:.4f} ({acc*100:.2f}%)")
    
    # 打印详细报告
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, digits=4))

    # 5. 结果可视化
    # 绘制混淆矩阵
    plot_confusion_matrix(best_model, X_test, y_test)
    
    # 展示几个实际的图片和预测结果
    show_prediction_examples(X_test, y_test, y_pred)

if __name__ == "__main__":
    main()
