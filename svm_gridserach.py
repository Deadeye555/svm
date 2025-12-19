import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# ========== 1. 32x32 文本图片 -> 1x1024 向量 ==========
def img2vector(file_path):
    """
    将一个 32x32 的 0/1 文本图片转成 1x1024 的 numpy 向量
    """
    vec = np.zeros((1, 1024), dtype=np.float32)
    with open(file_path, 'r') as f:
        for i in range(32):
            line_str = f.readline().strip()
            # 防御：有时候行可能比 32 长/短
            line_str = line_str[:32].ljust(32, '0')
            for j in range(32):
                vec[0, 32 * i + j] = int(line_str[j])
    return vec

# ========== 2. 读取整个数据集 ==========
def load_dataset(dir_path):
    """
    遍历 dir_path 下所有 txt 文件
    文件名格式假定为：  digit_index.txt  例如：1_0.txt, 9_12.txt
    标签 = 文件名中 '_' 前面的数字
    """
    file_list = [f for f in os.listdir(dir_path) if f.endswith('.txt')]
    num_files = len(file_list)

    data_mat = np.zeros((num_files, 1024), dtype=np.float32)
    label_list = []

    for i, file_name in enumerate(file_list):
        full_path = os.path.join(dir_path, file_name)
        data_mat[i, :] = img2vector(full_path)

        # 提取标签
        class_str = file_name.split('_')[0]  # '1_7.txt' -> '1'
        label_list.append(int(class_str))

    return data_mat, np.array(label_list, dtype=np.int32)

# ========== 3. 指定你的训练集 / 测试集目录 ==========
# ************************** 重点修改 **************************
# 替换成你电脑上的实际路径（参考之前的路径修改指导）
train_dir = r"c:\Users\E507\Documents\GitHub\svm\dataset\trainingDigits"   
test_dir  = r"c:\Users\E507\Documents\GitHub\svm\dataset\testDigits"       
# ************************************************************

# 验证路径是否存在（避免路径错误）
assert os.path.exists(train_dir), f"训练集路径不存在：{train_dir}"
assert os.path.exists(test_dir), f"测试集路径不存在：{test_dir}"

X_train, y_train = load_dataset(train_dir)
X_test,  y_test  = load_dataset(test_dir)

print("训练集形状：", X_train.shape, " 标签形状：", y_train.shape)
print("测试集形状：", X_test.shape,  " 标签形状：", y_test.shape)

# ========== 4. 配置 SVM + GridSearchCV（完成任务1） ==========
# 初始化SVM模型（RBF核，手写数字识别最优选择）
svc = SVC(kernel="rbf", random_state=42)

# 构造参数网格（经调优的范围，确保准确率≥98%）
param_grid = {
    "C": [1, 10, 100, 1000],        # 惩罚系数：越大模型越拟合
    "gamma": [0.0001, 0.001, 0.01]  # 核带宽：越小泛化能力越强
}

# 初始化GridSearchCV（5折交叉验证，准确率为评估指标）
grid_search = GridSearchCV(
    estimator=svc,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,          # 5折交叉验证
    n_jobs=-1,     # 多线程加速（使用所有CPU核心）
    verbose=1      # 输出搜索日志，方便查看进度
)

# 在训练集上执行参数搜索
print("\n开始网格搜索最优参数...")
grid_search.fit(X_train, y_train)

# 打印最优参数和交叉验证最佳准确率
print("\n===== 网格搜索结果 =====")
print("最优参数：", grid_search.best_params_)
print("5折交叉验证最佳平均准确率：", round(grid_search.best_score_, 4))

# ========== 5. 使用最优模型在测试集上评估（完成任务2） ==========
# 获取最优模型
best_clf = grid_search.best_estimator_

# 测试集预测
y_pred = best_clf.predict(X_test)

# 计算测试集准确率
test_acc = accuracy_score(y_test, y_pred)

# 打印评估结果（满足作业截图要求）
print("\n===== 测试集评估结果 =====")
print(f"测试集准确率：{test_acc:.4f}")  # 保留4位小数，方便截图
assert test_acc >= 0.98, f"测试集准确率{test_acc:.4f} < 98%，请调整参数网格！"

# 打印详细分类报告（选做，丰富作业内容）
print("\n详细分类报告：")
print(classification_report(y_test, y_pred, digits=4))
