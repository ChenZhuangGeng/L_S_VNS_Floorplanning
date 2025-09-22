import warnings
import os
import sys
import math
import pickle
import time
import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader
import torch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

# 屏蔽不必要的警告
warnings.filterwarnings("ignore")

# 动态添加PyTorch库路径 (主要用于Windows)
try:
    import torch

    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if sys.platform == 'win32' and os.path.isdir(torch_lib_path):
        os.add_dll_directory(torch_lib_path)
except (ImportError, FileNotFoundError, AttributeError):
    pass


def set_global_seed(seed):
    """设置项目中所有主要随机源的种子，以确保实验的可复现性。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class GCN(torch.nn.Module):
    """定义的图卷积网络模型。"""

    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super(GCN, self).__init__()
        torch.manual_seed(520)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear1 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.linear2 = torch.nn.Linear(hidden_channels // 2, hidden_channels // 4)
        self.linear3 = torch.nn.Linear(hidden_channels // 4, out_channels)

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def train(loader, model, optimizer, criterion, device):
    """模型的训练函数，返回训练集的MSE和MAE。"""
    model.train()
    total_mse_loss, total_mae_loss = 0, 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss_mse = criterion(out.view(-1), data.y)
        loss_mse.backward()
        optimizer.step()
        total_mse_loss += loss_mse.item() * data.num_graphs
        with torch.no_grad():
            loss_mae = torch.nn.functional.l1_loss(out.view(-1), data.y)
            total_mae_loss += loss_mae.item() * data.num_graphs
    avg_mse = total_mse_loss / len(loader.dataset)
    avg_mae = total_mae_loss / len(loader.dataset)
    return avg_mse, avg_mae


def test(loader, model, criterion, device):
    """模型的测试/评估函数，返回验证集的MSE和MAE。"""
    model.eval()
    total_mse_loss, total_mae_loss = 0, 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss_mse = criterion(out.view(-1), data.y)
            total_mse_loss += loss_mse.item() * data.num_graphs
            loss_mae = torch.nn.functional.l1_loss(out.view(-1), data.y)
            total_mae_loss += loss_mae.item() * data.num_graphs
    avg_mse = total_mse_loss / len(loader.dataset)
    avg_mae = total_mae_loss / len(loader.dataset)
    return avg_mse, avg_mae


def get_cosine_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
                                    num_cycles: float = 0.5, last_epoch: int = -1):
    """创建学习率调度器。"""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def run_experiment(args):
    """针对单个数据文件，执行完整的模型训练、评估和结果保存流程。"""
    # --- 0. 初始化和路径设置 ---
    set_global_seed(520)
    file_basename = os.path.splitext(os.path.basename(args.input_file))[0]
    output_path = os.path.join(args.output_dir, file_basename)
    os.makedirs(output_path, exist_ok=True)

    print(f"\n{'=' * 25}")
    print(f"开始处理数据集: {args.input_file}")
    print(f"所有结果将保存至: {output_path}")
    print(f"{'=' * 25}\n")

    # --- 1. 数据加载和预处理 ---
    try:
        with open(args.input_file, 'rb') as file:
            x_train_raw, x_test_raw, y_train_raw, y_test_raw = pickle.load(file)
    except FileNotFoundError:
        print(f"[错误] 数据文件不存在: {args.input_file}。已跳过。")
        return

    train_dataset = [Data(x=data.x, edge_index=data.edge_index, y=torch.tensor([label], dtype=torch.float)) for
                     data, label in zip(x_train_raw, y_train_raw)]
    test_dataset = [Data(x=data.x, edge_index=data.edge_index, y=torch.tensor([label], dtype=torch.float)) for
                    data, label in zip(x_test_raw, y_test_raw)]

    if not train_dataset:
        print("[错误] 训练数据集为空，无法继续。")
        return

    feature_len = train_dataset[0].num_features
    print(f"特征维度: {feature_len}, 训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

    # --- 2. 创建 DataLoader ---
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # --- 3. 模型、损失函数、优化器和设备设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")
    model = GCN(in_channels=feature_len, hidden_channels=args.hidden_channels).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=args.epochs)

    patience = args.patience
    epochs_no_improve = 0
    print(f"开始训练... 最大轮次: {args.epochs}, 早停耐心值: {patience}")

    # --- 4. 训练循环 ---
    train_mse_history, val_mse_history = [], []
    train_mae_history, val_mae_history = [], []
    min_val_mse = float('inf')
    mae_of_best_model = float('inf')
    best_epoch = -1
    model_save_path = os.path.join(output_path, f'{file_basename}_model.pt')

    training_start_time = time.time()
    for epoch in range(args.epochs):
        train_mse, train_mae = train(train_loader, model, optimizer, criterion, device)
        val_mse, val_mae = test(test_loader, model, criterion, device)
        scheduler.step()

        train_mse_history.append(train_mse)
        val_mse_history.append(val_mse)
        train_mae_history.append(train_mae)
        val_mae_history.append(val_mae)

        # 模型的保存与早停逻辑
        if val_mse < min_val_mse:
            min_val_mse = val_mse
            mae_of_best_model = val_mae
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            # 当找到更优模型时，打印一条简洁的提示
            print(f"Epoch {epoch:03d}: 验证MSE损失降低. 模型已保存。")
        else:
            epochs_no_improve += 1

        # CHANGED: 恢复您期望的逐轮打印格式
        print(
            f'Epoch:{epoch:03d} | Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f} | 最佳模型 (Epoch {best_epoch}): MSE={min_val_mse:.6f}, MAE={mae_of_best_model:.6f}'
        )

        # 检查是否需要早停
        if epochs_no_improve >= patience:
            print(f"\n连续 {patience} 个Epoch验证损失未改善，触发早停机制！")
            break

    print(f"\n训练总用时: {time.time() - training_start_time:.2f} s")
    print(f"最佳验证MSE: {min_val_mse:.6f} (其对应MAE为 {mae_of_best_model:.6f}) 出现在第 {best_epoch} 个epoch。")

    # --- 5. 结果可视化 ---
    # (这部分代码无需改动)
    # 图1：MSE损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_mse_history, label="Train Loss (MSE)")
    plt.plot(val_mse_history, label="Validation Loss (MSE)")
    if best_epoch != -1:
        plt.axvline(best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
        plt.axhline(min_val_mse, color='g', linestyle='--', label=f'Min Val MSE ({min_val_mse:.4f})')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, 'IterativeImg_MSE.png'), dpi=300)
    plt.close()

    # 图2：MAE指标曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_mae_history, label="Train MAE")
    plt.plot(val_mae_history, label="Validation MAE")
    if best_epoch != -1:
        plt.axvline(best_epoch, color='r', linestyle='--', label=f'Best Epoch (based on MSE)')
        plt.axhline(mae_of_best_model, color='g', linestyle='--',
                    label=f'Val MAE at Best Epoch ({mae_of_best_model:.4f})')
    plt.xlabel("Epoch")
    plt.ylabel("MAE (Mean Absolute Error)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, 'IterativeImg_MAE.png'), dpi=300)
    plt.close()
    print("MSE和MAE的收敛曲线图均已保存。")

    # --- 6. 最终评估 ---
    # (这部分代码无需改动)
    print("\n使用最佳模型进行最终评估...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            y_pred_list.append(out.view(-1).cpu().numpy())
            y_true_list.append(data.y.cpu().numpy())
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    final_mse = mean_squared_error(y_true, y_pred)
    final_rmse = np.sqrt(final_mse)
    final_mae = mean_absolute_error(y_true, y_pred)
    final_mape = mean_absolute_percentage_error(y_true, y_pred)

    print("最终评估结果:")
    print(f"  均方误差 (MSE): {final_mse:.6f}")
    print(f"  均方根误差 (RMSE): {final_rmse:.6f}")
    print(f"  平均绝对误差 (MAE): {final_mae:.6f}")
    print(f"  平均绝对百分比误差 (MAPE): {final_mape:.6f}")

    # 真实值 vs 预测值散点图
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Ideal Case (y=x)")
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'PredictShow.png'), dpi=300)
    plt.close()

    print(f"\n结果图表已保存到 '{output_path}' 目录。")
    print(f"--- 完成处理: {os.path.basename(args.input_file)} ---\n")


# --- 程序主入口 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GCN 模型训练与评估脚本")
    parser.add_argument('-i', '--input_file', type=str, required=True,
                        help='输入的pickle数据文件路径 (例如: data/ALLinONE/SampleAll_gcn_data.txt)')
    parser.add_argument('-o', '--output_dir', type=str, default='results', help='保存结果的根目录')
    parser.add_argument('--epochs', type=int, default=100000, help='最大训练轮次（默认10万）')
    parser.add_argument('--patience', type=int, default=10000, help='早停的耐心值（默认10万）')
    parser.add_argument('--batch_size', type=int, default=2048, help='训练和测试的批量大小')
    parser.add_argument('--hidden_channels', type=int, default=1024, help='GCN模型的隐藏层维度')
    # CHANGED: 移除了 print_every 参数，因为现在每轮都会打印

    args = parser.parse_args()

    overall_start_time = time.time()
    run_experiment(args)
    overall_end_time = time.time()

    print(f"\n{'=' * 25}")
    print("所有任务已完成！")
    print(f"总耗时: {(overall_end_time - overall_start_time) / 60:.2f} 分钟。")
    print(f"{'=' * 25}")