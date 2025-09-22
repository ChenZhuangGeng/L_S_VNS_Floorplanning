# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/8/31 10:57
from datetime import datetime
import sys
# sys.path.append('/home/eda220806/EDA_Last')
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from torch_geometric.nn import TopKPooling, GCNConv, global_mean_pool

from src.function import ReadFileUtils
from src.model.ClassificationAndRegression.EvaluateModel import *
import random
import torch
import os

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(520)
        self.hidden_channels = hidden_channels
        self.conv1 = GCNConv(28, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.top_k_pooling = TopKPooling(hidden_channels, ratio=0.8)
        self.linear1 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.linear2 = torch.nn.Linear(hidden_channels // 2, hidden_channels // 4)
        self.linear3 = torch.nn.Linear(hidden_channels // 4, 1)
        self.linear = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        # 特征提取
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)

        # 回归器
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x




class VNSAlgorithm_C_R:
    # 构造函数
    def __init__(self, instance):
        self.instance = Instance.copy(instance)  # 实例对象
        self.items = self.instance.items  # 模块列表（列表）
        self.links = self.instance.links  # 连接列表（列表）
        self.rule = self.instance.rule  # 规则字典（字典）
        self.area = self.instance.area  # 布线区域（顶点坐标列表）
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 2. 从当前目录向上回退3级找到项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

        # 3. 构建模型的完整路径
        self.model_name = os.path.join(
            project_root,
            'src',
            'model',
            'ScoreForPlan',
            'score_ml_gcn_batch2048_model.pt'
        )

    # 算法求解外部接口
    def solve(self, init_sequence, timer, min_x, min_y, forGCNTrain_txt_output_path):
        # 算法参数
        self.local_cnt = 500  # 局部搜索次数 500
        self.y = []
        self.yy = []
        # 历史数据表的最大容量
        self.max_history_list_len = 10
        # 是否要开启测试二分类准确率模式
        self.test_classifier_model_enabled = False
        # 是否先进先出
        self.first_in_first_out_enabled = False
        # 是否开启二分类模型（如果不想用二分类就设置为False）
        self.classifier_switch = False
        if self.test_classifier_model_enabled is True and self.classifier_switch is False:
            raise RuntimeError("test_classifier_model_enabled=True时，classifier_switch必须也为True")
        # 机器学习分类模型是否被启动了(不用手动设置，纯粹代表当前二分类是否可用，即是否被训练过)
        self.classifier_enable = False
        # 记录训练数据集size的变化列表
        self.train_data_size_list = []
        # 卡时
        self.max_time = timer
        s = time.time()
        # 历史数据存放列表
        self.history_data_list = []
        # 历史预测错误的列表
        self.error_data_list = []
        # 机器学习预测打分模型
        # self.model_name = '/home/eda220806/EDA_Last/src/model/ScoreForPlan/score_ml_gcn_batch2048_model.pt'
        # self.model_name = r'D:\IDEA2020\Project\EDA_Last\src\model\ScoreForPlan\score_ml_gcn_batch2048_model.pt'

        self.device = torch.device("cpu")

        self.score_ml_model = torch.load(self.model_name)


        self.score_ml_model.to(self.device)

        # 统计VNS评估次数和评估的总时间，最后保存用
        self.vns_evaluate_cnt = 0
        evaluation_total_time = 0.0

        # 初始化
        self.x_data, self.y_data = [], []
        self.train_classifier_timer = 0  # 训练第一个分类模型用时
        start_eval_time = time.perf_counter()
        self.best_evaluation = self.evaluate(init_sequence, min_x, min_y, forGCNTrain_txt_output_path)  # 初始化最优解
        end_eval_time = time.perf_counter()
        evaluation_total_time += (end_eval_time - start_eval_time)  # 直接累加秒数
        self.predict_cnt = 0  # 预测的次数
        self.fit_cnt = 0  # 训练的次数
        self.predict_true_cnt = 0  # 预测正确的次数
        # 开始算法的迭代
        while (time.time() - s) < self.max_time:
            local_sequence = self.swap_sequence(init_sequence.copy())
            start_eval_time = time.perf_counter()
            local_best_evaluation = self.evaluate(local_sequence, min_x, min_y, forGCNTrain_txt_output_path)
            end_eval_time = time.perf_counter()
            evaluation_total_time += (end_eval_time - start_eval_time)  # 直接累加秒数
            if local_best_evaluation.obj_value > self.best_evaluation.obj_value:
                self.best_evaluation = local_best_evaluation
            self.VND(local_best_evaluation, s, min_x, min_y, forGCNTrain_txt_output_path)

        epochs_size = 0
        if len(self.yy) != 0:
            self.y.append(self.yy)
        for yy in self.y:
            epochs_size += len(yy)
        # 迭代完毕，返回最佳结果
        best_obj_value = self.best_evaluation.obj_value
        timer = 0
        accuracy = self.predict_true_cnt / self.predict_cnt if self.predict_cnt > 0 else 0
        result_list_len = len(self.best_evaluation.result_list)


        return self.best_evaluation, MyResult(best_obj_value, timer, accuracy, epochs_size, self.fit_cnt,
                                              self.predict_cnt,
                                              self.predict_true_cnt, result_list_len), self.vns_evaluate_cnt, evaluation_total_time



    # 训练分类模型
    def train_classifier(self):
        # 构造数据
        t = time.time()
        self.x_data, self.y_data = [], []
        for i1, d1 in enumerate(self.history_data_list):
            for i2, d2 in enumerate(self.history_data_list):
                if i1 != i2:
                    x = d1[1].copy()
                    x.extend(d2[1])
                    self.x_data.append(x)
                    self.y_data.append(1 if d1[0] > d2[0] else 0)
        for y, x in self.error_data_list:
            self.x_data.append(x)
            self.y_data.append(y)
        # 机器学习二分类模型
        self.classifier = LGBMClassifier()
        # self.classifier = KNeighborsClassifier()
        # 训练模型
        self.classifier.fit(self.x_data, self.y_data)
        self.train_classifier_timer += (time.time() - t)

    # 判断序列是否存在于历史数据表中
    def in_history_data_list(self, sequence):
        for history_data in self.history_data_list:
            b = True
            for i in range(len(history_data[1])):
                if history_data[1][i] != sequence[i]:
                    b = False
                    break
            if b:
                return True
        return False

    # 根据序列摆放并评价
    def evaluate(self, sequence, min_x, min_y, forGCNTrain_txt_output_path):
        # em = EvaluateModel(sequence, self.items, self.links, self.rule, self.area, self.score_ml_model,
        #                  self.model_name, self.device).evaluate()
        # for result in em.result_list:
        #     print(result.to_string())
        #
        #
        # return em
        self.vns_evaluate_cnt += 1
        # 处理一下这个路径 由5-1改为5-1-1，第三个是评估次数self.vns_evaluate_cnt
        modified_path = ReadFileUtils.add_evaluation_count_to_path(forGCNTrain_txt_output_path, self.vns_evaluate_cnt)
        return EvaluateModel(sequence, self.items, self.links, self.rule, self.area, self.score_ml_model,
                         self.model_name, self.device).evaluate(min_x, min_y, modified_path)

    # 两两互换，获取新序列
    def swap_sequence(self, sequence):
        r1 = random.randint(0, len(sequence) - 1)
        r2 = random.randint(0, len(sequence) - 1)
        while r1 == r2:
            r2 = random.randint(0, len(sequence) - 1)
        temp = sequence[r1]
        sequence[r1] = sequence[r2]
        sequence[r2] = temp
        return sequence

    # 片段交换
    def rotate_sequence(self, sequence):
        split_index = random.randint(1, len(sequence) - 1)
        temp = sequence[split_index:]
        for i in sequence[:split_index]:
            temp.append(i)
        return temp

    def VND(self, local_best_evaluation, s, min_x, min_y, forGCNTrain_txt_output_path):
        t = 0
        n = 0
        while n < 2 and (time.time() - s) < self.max_time:
            if t > self.local_cnt:
                n += 1
                t = 0
                continue
            temp_sequence = []
            # 迭代次数自增
            self.yy.append(self.best_evaluation.obj_value)
            try:
                self.train_data_size_list.append(len(self.x_data))
            except:
                self.train_data_size_list.append(0)
            if n == 0:
                # 两两互换
                temp_sequence = self.swap_sequence(local_best_evaluation.sequence.copy())
            elif n == 1:
                # 片段交换
                temp_sequence = self.rotate_sequence(local_best_evaluation.sequence.copy())
            else:
                raise RuntimeError("不存在的邻域n:" + str(n))
            # 判断其是否存在于历史数据表，如果存在则重新随机
            if self.in_history_data_list(temp_sequence):
                t += 1
                continue
            # 如果机器学习分类模型被启用了，那么就进行分类预测，加快迭代速度
            if self.classifier_enable is True:
                x_data = temp_sequence.copy()
                x_data.extend(local_best_evaluation.sequence)
                y_hat = self.classifier.predict([x_data])
                if self.test_classifier_model_enabled is False:
                    # 如果预测为1，说明新生成的序列比当前最优解更好，则进行评价
                    if y_hat[0] == 1:
                        self.predict_cnt += 1
                        temp_evaluation = self.evaluate(temp_sequence, min_x, min_y, forGCNTrain_txt_output_path)
                        # 更新全局最优解
                        if temp_evaluation.obj_value > local_best_evaluation.obj_value:
                            local_best_evaluation = temp_evaluation
                            t = 0
                            n = 0
                            self.predict_true_cnt += 1
                            if local_best_evaluation.obj_value > self.best_evaluation.obj_value:
                                self.best_evaluation = local_best_evaluation
                        else:
                            # 说明预测错误,加入到错误列表
                            self.error_data_list.append([0, x_data])
                            x_data_2 = self.best_evaluation.sequence.copy()
                            x_data_2.extend(temp_sequence)
                            self.error_data_list.append([1, x_data_2])
                            # 更新迭代次数记录器
                            t += 1
                            if t >= self.local_cnt:
                                t = 0
                                n += 1
                                self.y.append(self.yy.copy())
                                self.yy = []
                            # 将新序列得分及其新序列加入历史数据表，并重新训练二分类模型
                        if self.classifier_switch is True:
                            if len(self.history_data_list) < self.max_history_list_len:
                                self.history_data_list.append(
                                    [temp_evaluation.obj_value, temp_evaluation.sequence.copy()])
                            else:
                                if self.first_in_first_out_enabled is True:
                                    self.history_data_list.pop(0)
                                self.history_data_list.append(
                                    [temp_evaluation.obj_value, temp_evaluation.sequence.copy()])
                                self.train_classifier()
                                self.fit_cnt += 1
                                self.classifier_enable = True
                    else:
                        # 否则继续迭代
                        t += 1
                        if t >= self.local_cnt:
                            t = 0
                            n += 1
                            self.y.append(self.yy.copy())
                            self.yy = []
                else:
                    self.predict_cnt += 1
                    temp_evaluation = self.evaluate(temp_sequence, min_x, min_y, forGCNTrain_txt_output_path)
                    # 更新全局最优解
                    if temp_evaluation.obj_value > local_best_evaluation.obj_value:
                        local_best_evaluation = temp_evaluation
                        t = 0
                        n = 0
                        if local_best_evaluation.obj_value > self.best_evaluation.obj_value:
                            self.best_evaluation = local_best_evaluation
                        if abs(y_hat[0] - 1) <= 1e-06:
                            self.predict_true_cnt += 1
                        else:
                            # 说明预测错误,加入到错误列表
                            self.error_data_list.append([1, x_data])
                            x_data_2 = self.best_evaluation.sequence.copy()
                            x_data_2.extend(temp_sequence)
                            self.error_data_list.append([0, x_data_2])
                    else:
                        if abs(y_hat[0] - 0) <= 1e-06:
                            self.predict_true_cnt += 1
                        else:
                            # 说明预测错误,加入到错误列表
                            self.error_data_list.append([0, x_data])
                            x_data_2 = self.best_evaluation.sequence.copy()
                            x_data_2.extend(temp_sequence)
                            self.error_data_list.append([1, x_data_2])
                    # 将新序列得分及其新序列加入历史数据表，并重新训练二分类模型
                    if self.classifier_switch is True:
                        if len(self.history_data_list) < self.max_history_list_len:
                            self.history_data_list.append([temp_evaluation.obj_value, temp_evaluation.sequence.copy()])
                        else:
                            if self.first_in_first_out_enabled is True:
                                self.history_data_list.pop(0)
                            self.history_data_list.append([temp_evaluation.obj_value, temp_evaluation.sequence.copy()])
                            self.train_classifier()
                            self.fit_cnt += 1
                            self.classifier_enable = True
            else:
                temp_evaluation = self.evaluate(temp_sequence, min_x, min_y, forGCNTrain_txt_output_path)
                # 将新序列得分及其新序列加入历史数据表，并重新训练二分类模型
                if self.classifier_switch is True:
                    if len(self.history_data_list) < self.max_history_list_len:
                        self.history_data_list.append([temp_evaluation.obj_value, temp_evaluation.sequence.copy()])
                    else:
                        if self.first_in_first_out_enabled is True:
                            self.history_data_list.pop(0)
                        self.history_data_list.append([temp_evaluation.obj_value, temp_evaluation.sequence.copy()])
                        self.train_classifier()
                        self.fit_cnt += 1
                        self.classifier_enable = True
                # 更新全局最优解
                if temp_evaluation.obj_value > local_best_evaluation.obj_value:
                    local_best_evaluation = temp_evaluation
                    t = 0
                    n = 0
                    if local_best_evaluation.obj_value > self.best_evaluation.obj_value:
                        self.best_evaluation = local_best_evaluation
                else:
                    t += 1
                    if t >= self.local_cnt:
                        t = 0
                        n += 1
                        self.y.append(self.yy.copy())
                        self.yy = []
