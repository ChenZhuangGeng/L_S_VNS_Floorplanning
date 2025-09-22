import copy
import sys
from tokenize import String

from src.function import ReadFileUtils
from src.function.RoutingScore import RoutingScore
from src.function.RoutingScore_LAHC_V4 import RoutingScore_Interface, DataUtils

sys.path.append('/home/eda220806/project/code')
import functools
import math
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from src.classes.EdaClasses import *
import src.function.WskhFunction as WskhFunction


class MyDataSet(Dataset):
    def __init__(self, x):
        self.data = x

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class EvaluateModel:
    def __init__(self, sequence, items, links, rule, area, score_ml_model, model_name, device):
        self.sequence = sequence  # 序列
        self.items = items  # 模块列表（列表）
        self.links = links  # 连接列表（列表）
        self.rule = rule  # 规则字典（字典）
        self.area = area  # 布线区域（顶点坐标列表）
        self.score_ml_model = score_ml_model  # 机器学习打分预测模型
        self.model_name = model_name  # 机器学习算法名字
        self.device = device  # GPU/CPU

    # 评价函数，根据序列评价
    def evaluate(self, min_x, min_y, start_time, max_time, GCN_ON, forGCNTrain_txt_output_path=None):
        # 参数
        # 误差
        self.error = 1e-06
        # 是否开启剩余空间能否放置矩形的判断
        self.waste_space_judge_place_rect_able = True
        # 初始化已放置模块的状态列表
        self.item_flags = [0 for i in range(len(self.sequence))]
        # 记录更新天际线的耗时
        self.timer = 0
        # 计算边界的所有水平线段(x,y,l)
        self.area_horizontal_line_list = WskhFunction.calc_horizontal_line_list(self.area)
        # 计算边界的所有垂直线段(x,y,l)
        self.area_vertical_line_list = WskhFunction.calc_vertical_line_list(self.area)
        # 初始化天际线列表
        sky_line_list = self.init_sky_line_list(self.area)
        # 初始化已放入模块列表
        result_list = []
        # 初始化已放入模块的面积
        area_sum = 0.0
        # 在这里初始化你的循环计数器,起码跑一次while循环才去判断有没有超时
        loop_count = 0
        # 开始循环
        while len(result_list) < len(self.sequence) and len(sky_line_list) > 0:

            # # ==================== 检查点 1 (主循环开始时) ====================
            # # 检查时间是否已经超限，如果超了就直接跳出主循环
            # if loop_count > 0 and start_time and max_time and (time.time() - start_time >= max_time):
            #     print(f"在第 {loop_count + 1} 次布局尝试前超时，提前退出。")
            #     break
            # # ===============================================================

            # print("-----------------------------")
            # 如果天际线长度为0，则直接返回None
            if sky_line_list[0][2] <= 0:
                sky_line_list.pop(0)
                continue
            # 初始化最大得分对象
            global_best_score = None
            best_index = -1
            self.hl_hr = self.get_h1_h2_by_sky_line(sky_line_list[0], sky_line_list)
            # 遍历所有L/T型模块，选出最高得分L/T型模块
            for i in range(len(self.sequence)):

                # # ==================== 检查点 2 (推荐! 细粒度控制) ====================
                # # 在尝试每个新模块前检查时间，这里的检查频率更高，控制更精确
                # if loop_count > 0 and start_time and max_time and (time.time() - start_time >= max_time):
                #     break  # 跳出内层 for 循环
                # # ===============================================================

                # 获取当前要放置的模块
                index = self.sequence[i]
                item = self.items[index]
                # 如果当前模块已经放入，则进入下一个块的探索
                if self.item_flags[index] == 1 or len(item.boundary) == 4:
                    continue
                # 遍历所有旋转情况
                for rotate_item in item.rotate_items:
                    score = self.score(item, rotate_item, result_list, sky_line_list.copy())
                    if score is not None and (
                            global_best_score is None or global_best_score.waste_value > score.waste_value or (
                            abs(global_best_score.waste_value - score.waste_value) <= self.error and self.calc_placeS_linkS_S(
                        global_best_score.deta_value, global_best_score.link_value) < self.calc_placeS_linkS_S(
                        score.deta_value, score.link_value))):
                        global_best_score = score
                        best_index = index
            # 如果L/T型的模块没找到最优的，则开始遍历矩形，找矩形里的最高得分
            if global_best_score is None or global_best_score.waste_value > 0:
                for index in range(len(self.sequence)):

                    # # ==================== 检查点 3 (同检查点2) ====================
                    # if loop_count > 0 and start_time and max_time and (time.time() - start_time >= max_time):
                    #     break  # 跳出内层 for 循环
                    # # =============================================================

                    # 获取当前要放置的模块
                    item = self.items[index]
                    # 如果当前模块已经放入，则进入下一个块的探索
                    if self.item_flags[index] == 1 or len(item.boundary) != 4:
                        continue
                    # 遍历所有旋转情况
                    for rotate_item in item.rotate_items:
                        score = self.score(item, rotate_item, result_list, sky_line_list.copy())
                        if score is not None and (
                                global_best_score is None or self.calc_placeS_linkS_S(global_best_score.rect_value,
                                                                                      global_best_score.link_value) < self.calc_placeS_linkS_S(
                            score.rect_value, score.link_value)):
                            global_best_score = score
                            best_index = index
            # 更新天际线、已放入模块列表和已放入模块的面积
            if global_best_score is not None:
                # print(len(result_list), global_best_score.result.item.name, global_best_score.to_string())
                sky_line_list = global_best_score.sky_line_list
                result_list.append(global_best_score.result)
                area_sum += global_best_score.result.item.area
                self.item_flags[best_index] = 1
            else:
                # 记录原天际线列表的长度
                old_sky_line_cnt = len(sky_line_list)
                # 当前天际线无法放置任何一个模块，故对其进行上移合并处理
                sky_line_list = self.up_move_sky_line(sky_line_list)
                # 如果不能上移合并，则删除这个天际线
                if len(sky_line_list) == old_sky_line_cnt:
                    sky_line_list.pop(0)
                if len(sky_line_list) == 0:
                    break
            # 对天际线列表重新进行排序
            sky_line_list = self.sort_sky_line_list_by_y_x(sky_line_list)
            # 在循环的末尾，增加计数器的值
            loop_count += 1
        # 返回评价结果
        s = time.time()
        boundary_list = []
        links, name_list, ports = [], [], []
        for link in self.links:
            links.append(link.link_dict)
        for result in result_list:
            name_list.append(result.rotate_item.name)
            ports.append(result.rotate_item.ports)
            boundary_list.append(
                self.get_absolute_boundary_by_sky_line(result.rotate_item.boundary.copy(), result.sky_line))
        # 预测得分


        # todo 实现布线算法返回布局布线过后的得分
        ## 输入: result_list, self.area, links, name_list, ports
        ## 输出: link_score
        if GCN_ON == False:
            items = []
            if len(result_list) <= 5:
                dis = 26
            elif len(result_list) <= 10:
                dis = 30
            elif len(result_list) <= 25:
                dis = 38
            elif len(result_list) <= 50:
                dis = 40
            else:
                dis = 45
            for result in result_list:
                # print("===result:", result.to_string(), "===")

                # 对模块进行缩小，并还原坐标
                copy_rsult =  copy.deepcopy(result)


                copy_rsult.rotate_item.boundary = WskhFunction.zoom_use_exhaustive_method(copy_rsult.rotate_item, -dis)
                copy_rsult.rotate_item.init_rotate_item()

                # 实际用
                copy_rsult.left_bottom_point = [copy_rsult.left_bottom_point[0] + dis / 2.0 + min_x,
                                            copy_rsult.left_bottom_point[1] + dis / 2.0 + min_y]

                item = copy_rsult.item  # 获取 item 对象的引用
                item.boundary = copy_rsult.rotate_item.boundary
                item.ports = copy_rsult.rotate_item.ports
                dx = copy_rsult.left_bottom_point[0]
                dy = copy_rsult.left_bottom_point[1]

                # 更新 item.boundary
                # 创建一个包含新坐标元组的新列表，并将其赋值回 item.boundary
                item.boundary = [(x + dx, y + dy) for x, y in item.boundary]

                new_ports = []
                for port_coord_list in item.ports:
                    updated_port_coord_list = [(x + dx, y + dy) for x, y in port_coord_list]
                    new_ports.append(updated_port_coord_list)
                item.ports = new_ports

                # print("===item:", item.to_stringbyczg(), "===")
                items.append(item)

            # 此处传的是result_list，具体还需要转化
            loc_area = [(x + min_x, y + min_y) for x, y in self.area]

            # items是全部的item集合，但是不一定全部都放得下所以需要对links进行过滤，传进去算法的links是过滤后的
            # 传old_links是为了计算all_items_count 为了评分的时候能知道总共有多少个模块
            real_links = DataUtils().filter_invalid_links(links, items)
            time_route, score = RoutingScore_Interface.Solve_Interface(items, loc_area, real_links, links, start_time, max_time)

            # print("===time_route:", time_route, "===")
            # print("===score:", score, "===")

            # Todo: 评估一次就保存一次结果，供GCN训练
            # ReadFileUtils.generate_layoutForGCNTrain_txt(self.area, items, score, forGCNTrain_txt_output_path)

            return Evaluation(score, self.sequence, result_list)

        else:
            # 原本的得分预测方式
            x = WskhFunction.calc_features_for_gcn(boundary_list, self.area, links, name_list, ports)
            x_data = [Data(x=x[0], edge_index=x[1])]
            data_set = MyDataSet(x_data)
            data_loader = DataLoader(data_set, batch_size=1, shuffle=True)
            # print("===type(data_loader):", type(data_loader), "===")
            for data in data_loader:
                predict_score = self.score_ml_model(data.x, data.edge_index, data.batch)
                # print("===type(predict_score):", type(predict_score),"===")
                # print("===预测得分:", predict_score[0][0].item(),"===")
                return Evaluation(predict_score[0][0].item(), self.sequence, result_list)




    # 指定旋转类型，进行评分
    def score(self, item, rotate_item, result_list, sky_line_list):
        # 获取模块的顶点个数
        l = len(rotate_item.boundary)
        # 从天际线列表中取出第一个元素作为放置的天际线
        sky_line = sky_line_list[0]
        # 如果超出天际线长度，则直接返回None(这个判断只对矩形有用)
        if l == 4 and self.is_out_of_sky_line(rotate_item, sky_line):
            return None
        # 如果上面都通过了，说明该天际线可以放置当前模块，那么进行放置，并计算分数
        score = None
        if l == 4:
            # 矩形的评分
            score = self.calc_S_score(item, rotate_item, result_list, sky_line_list)
        elif l == 6:
            # L型的评分
            score = self.calc_L_score(item, rotate_item, result_list, sky_line_list)
        elif l == 8:
            # T型的评分
            score = self.calc_T_score(item, rotate_item, result_list, sky_line_list)
        else:
            raise RuntimeError("出现了未知形状的模块")
        return score

    # 计算位置评分和连线评分的加权评分
    def calc_placeS_linkS_S(self, placeS, linkS):
        # placeS的权重
        w = 0
        # 返回综合得分
        return (1 / (1 + math.exp(-placeS))) * w + (1 / (1 + math.exp(-linkS))) * (1 - w)  # Sigmoid

    # 计算T块的得分
    def calc_T_score(self, item, rotate_item, result_list, sky_line_list):
        # 复制天际线列表
        copy_sky_line_list = sky_line_list.copy()
        # 记录原来天际线的个数
        old_sky_line_cnt = len(copy_sky_line_list)
        # 从天际线列表中取出第一个元素作为放置的天际线
        sky_line = copy_sky_line_list.pop(0)
        # ======================================================== 0度，往左放 ========================================================
        if rotate_item.type == '0' and rotate_item.w <= sky_line[2]:
            left_score = self.place_left_for_L_T(sky_line, item, rotate_item, copy_sky_line_list.copy(),
                                                 old_sky_line_cnt,
                                                 rotate_item.orient, result_list)
            right_score = self.place_right_for_L_T(sky_line, item, rotate_item, copy_sky_line_list.copy(),
                                                   old_sky_line_cnt,
                                                   rotate_item.orient, result_list)
            if left_score is None:
                return right_score
            if right_score is None:
                return left_score
            return left_score if left_score.deta_value > right_score.deta_value else right_score
        elif rotate_item.type == '90' and rotate_item.w1 <= sky_line[2]:
            # ======================================================== 90度，往右放 ========================================================
            left_bottom_point = (sky_line[0] + sky_line[2] - rotate_item.w1, sky_line[1])
            # 如果超出边界，则直接返回None
            if self.is_out_of_bound(rotate_item, self.area,
                                    left_bottom_point):
                return None
            # 需要对其进行重叠判断，如果重叠则不能在这个点进行放置，直接返回None
            if self.is_overlap(result_list, rotate_item, left_bottom_point):
                return None
            # 到这里说明可以放
            # 对天际线列表重新进行排序
            copy_sky_line_list = self.sort_sky_line_list_by_x_y(copy_sky_line_list)
            # 找到右边的天际线进行切割，并计算浪费
            waste = 0
            # 找到第二高度
            b = False
            for i in range(len(copy_sky_line_list)):
                cs = copy_sky_line_list[i]
                if cs[0] == sky_line[0] + sky_line[2] and cs[1] >= sky_line[1]:
                    if cs[1] <= rotate_item.h3 + sky_line[1]:
                        # 上方浪费
                        if cs[2] < rotate_item.w - rotate_item.floor_horizontal_line[2]:
                            return None
                        # 判断浪费的空间里是否可以放下矩形,如果可以放下，则直接返回None
                        if self.waste_space_judge_place_rect_able is True and self.judge_waste_space_can_place_rect_able(
                                rotate_item.w - rotate_item.floor_horizontal_line[2],
                                rotate_item.h3 + sky_line[1] - cs[1]):
                            return None
                        copy_sky_line_list[i] = (cs[0] + (rotate_item.w - rotate_item.floor_horizontal_line[2]), cs[1],
                                                 cs[2] - (rotate_item.w - rotate_item.floor_horizontal_line[2]))
                        waste = (rotate_item.w - rotate_item.floor_horizontal_line[2]) * (
                                rotate_item.h3 + sky_line[1] - cs[1])
                    else:
                        # 下方浪费
                        # 判断浪费的空间里是否可以放下矩形,如果可以放下，则直接返回None
                        if self.waste_space_judge_place_rect_able is True and self.judge_waste_space_can_place_rect_able(
                                rotate_item.w - rotate_item.floor_horizontal_line[2], rotate_item.h3 + sky_line[1]):
                            return None
                        waste = (rotate_item.h3 + sky_line[1]) * (rotate_item.w - rotate_item.floor_horizontal_line[2])
                    b = True
                    if copy_sky_line_list[i][2] <= 0:
                        copy_sky_line_list.pop(i)
                    break
            if b is False:
                return None
            # 常规切割
            if sky_line[2] - rotate_item.floor_horizontal_line[2] > 0:
                copy_sky_line_list.append((sky_line[0], sky_line[1],
                                           sky_line[2] - rotate_item.floor_horizontal_line[2]))
            # 加入item潜在的天际线
            for add_sky_line in rotate_item.add_sky_line_list:
                new_add_sky_line = (
                    add_sky_line[0] + sky_line[0] + (sky_line[2] - rotate_item.floor_horizontal_line[2]),
                    add_sky_line[1] + sky_line[1], add_sky_line[2])
                for sk in copy_sky_line_list:
                    if sk[1] > new_add_sky_line[1] and str(sk) not in self.init_sky_line_set:
                        if sk[0] <= new_add_sky_line[0] and sk[0] + sk[2] >= new_add_sky_line[0]:
                            # 左边切
                            new_add_sky_line = (sk[0] + sk[2], new_add_sky_line[1],
                                                new_add_sky_line[0] + new_add_sky_line[2] - (sk[0] + sk[2]))
                            break
                        elif sk[0] >= new_add_sky_line[0] and sk[0] <= new_add_sky_line[0] + new_add_sky_line[2]:
                            # 右边切
                            new_add_sky_line = (new_add_sky_line[0], new_add_sky_line[1], sk[0] - new_add_sky_line[0])
                            break
                if new_add_sky_line[2] > 0:
                    copy_sky_line_list.append(new_add_sky_line)
            # 对天际线列表重新进行排序
            copy_sky_line_list = self.sort_sky_line_list_by_y_x(copy_sky_line_list)
            # 遍历新的天际线列表，将可以合并的天际线进行合并
            copy_sky_line_list = self.combine_sky_line(copy_sky_line_list)
            # 计算分数
            score = old_sky_line_cnt - len(copy_sky_line_list)
            # 模块中心点坐标
            center_position = self.calc_center_position(rotate_item, left_bottom_point)
            # 生成当前得分对象
            return Score(score, self.calc_link_score(center_position, rotate_item, result_list), 0, 0,
                         copy_sky_line_list,
                         Result(Item.copy(item), rotate_item, rotate_item.orient, center_position,
                                sky_line, left_bottom_point))
        elif rotate_item.type == '270' and rotate_item.floor_horizontal_line[2] <= sky_line[2]:
            # ======================================================== 270度，往左放 ========================================================
            left_bottom_point = (sky_line[0] - rotate_item.w2, sky_line[1])
            # 如果超出边界，则直接返回None
            if self.is_out_of_bound(rotate_item, self.area,
                                    left_bottom_point):
                return None
            # 存在重叠，则直接返回None
            # 说明对标点不在包络矩形的左下角，所以是特殊的L型，所以在放置时，需要对其进行重叠判断，如果重叠则不能在这个点进行放置
            if self.is_overlap(result_list, rotate_item, left_bottom_point):
                return None
            # 到这里说明可以放
            # 对天际线列表重新进行排序
            copy_sky_line_list = self.sort_sky_line_list_by_x_y(copy_sky_line_list)
            # 找到左边的天际线进行切割，并计算浪费
            waste = 0
            # 找到第二高度
            b = False
            for i in range(len(copy_sky_line_list)):
                cs = copy_sky_line_list[i]
                if cs[0] + cs[2] == sky_line[0] and cs[1] >= sky_line[1]:
                    if cs[1] <= rotate_item.h1 + sky_line[1]:
                        # 上方浪费
                        if cs[2] < rotate_item.w - rotate_item.floor_horizontal_line[2]:
                            return None
                        # 判断浪费的空间里是否可以放下矩形,如果可以放下，则直接返回None
                        if self.waste_space_judge_place_rect_able is True and self.judge_waste_space_can_place_rect_able(
                                rotate_item.w - rotate_item.floor_horizontal_line[2],
                                rotate_item.h1 + sky_line[1] - cs[1]):
                            return None
                        copy_sky_line_list[i] = (
                            cs[0], cs[1], cs[2] - (rotate_item.w - rotate_item.floor_horizontal_line[2]))
                        waste = (rotate_item.w - rotate_item.floor_horizontal_line[2]) * (
                                rotate_item.h1 + sky_line[1] - cs[1])
                    else:
                        # 下方浪费
                        # 判断浪费的空间里是否可以放下矩形,如果可以放下，则直接返回None
                        if self.waste_space_judge_place_rect_able is True and self.judge_waste_space_can_place_rect_able(
                                rotate_item.w - rotate_item.floor_horizontal_line[2], rotate_item.h1 + sky_line[1]):
                            return None
                        waste = (rotate_item.h1 + sky_line[1]) * (rotate_item.w - rotate_item.floor_horizontal_line[2])
                    b = True
                    if copy_sky_line_list[i][2] <= 0:
                        copy_sky_line_list.pop(i)
                    break
            if b is False:
                return None
            # 常规切割
            if sky_line[2] - rotate_item.floor_horizontal_line[2] > 0:
                copy_sky_line_list.append((sky_line[0] + rotate_item.floor_horizontal_line[2], sky_line[1],
                                           sky_line[2] - rotate_item.floor_horizontal_line[2]))
            # 加入item潜在的天际线
            for add_sky_line in rotate_item.add_sky_line_list:
                new_add_sky_line = (
                    add_sky_line[0] + sky_line[0] - rotate_item.w2,
                    add_sky_line[1] + sky_line[1], add_sky_line[2])
                for sk in copy_sky_line_list:
                    if sk[1] > new_add_sky_line[1] and str(sk) not in self.init_sky_line_set:
                        if sk[0] <= new_add_sky_line[0] and sk[0] + sk[2] >= new_add_sky_line[0]:
                            # 左边切
                            new_add_sky_line = (sk[0] + sk[2], new_add_sky_line[1],
                                                new_add_sky_line[0] + new_add_sky_line[2] - (sk[0] + sk[2]))
                            break
                        elif sk[0] >= new_add_sky_line[0] and sk[0] <= new_add_sky_line[0] + new_add_sky_line[2]:
                            # 右边切
                            new_add_sky_line = (new_add_sky_line[0], new_add_sky_line[1], sk[0] - new_add_sky_line[0])
                            break
                if new_add_sky_line[2] > 0:
                    copy_sky_line_list.append(new_add_sky_line)
            # 对天际线列表重新进行排序
            copy_sky_line_list = self.sort_sky_line_list_by_y_x(copy_sky_line_list)
            # 遍历新的天际线列表，将可以合并的天际线进行合并
            copy_sky_line_list = self.combine_sky_line(copy_sky_line_list)
            # 计算分数
            score = old_sky_line_cnt - len(copy_sky_line_list)
            # 模块中心点坐标
            center_position = self.calc_center_position(rotate_item, left_bottom_point)
            # 生成当前得分对象
            return Score(score, self.calc_link_score(center_position, rotate_item, result_list), 0, 0,
                         copy_sky_line_list,
                         Result(Item.copy(item), rotate_item, rotate_item.orient, center_position,
                                sky_line, left_bottom_point))
        elif len(rotate_item.add_sky_line_list) == 1:
            # ======================================================== 180度，刚好放下 ========================================================
            # T型180度，刚好放下才放
            if sky_line[2] == rotate_item.w2:
                hl, hr = self.hl_hr
                if hl == rotate_item.h2 and hr == rotate_item.h3:
                    left_bottom_point = (sky_line[0] - rotate_item.w1, sky_line[1])
                    # 如果超出边界，则直接返回None
                    if self.is_out_of_bound(rotate_item, self.area,
                                            left_bottom_point):
                        return None
                    # 存在重叠，则直接返回None
                    # 说明对标点不在包络矩形的左下角，所以是特殊的L型，所以在放置时，需要对其进行重叠判断，如果重叠则不能在这个点进行放置
                    if self.is_overlap(result_list, rotate_item, left_bottom_point):
                        return None
                    # 常规切割（这里不需要常规切割，因为正好放入，所以当前天际线更新后长度为0，被删除）

                    # 加入item潜在的天际线
                    for add_sky_line in rotate_item.add_sky_line_list:
                        new_add_sky_line = (
                            add_sky_line[0] + sky_line[0] - rotate_item.w1,
                            add_sky_line[1] + sky_line[1], add_sky_line[2])
                        for sk in copy_sky_line_list:
                            if sk[1] > new_add_sky_line[1] and str(sk) not in self.init_sky_line_set:
                                if sk[0] <= new_add_sky_line[0] and sk[0] + sk[2] >= new_add_sky_line[0]:
                                    # 左边切
                                    new_add_sky_line = (sk[0] + sk[2], new_add_sky_line[1],
                                                        new_add_sky_line[0] + new_add_sky_line[2] - (sk[0] + sk[2]))
                                    break
                                elif sk[0] >= new_add_sky_line[0] and sk[0] <= new_add_sky_line[0] + new_add_sky_line[
                                    2]:
                                    # 右边切
                                    new_add_sky_line = (
                                        new_add_sky_line[0], new_add_sky_line[1], sk[0] - new_add_sky_line[0])
                                    break
                        if new_add_sky_line[2] > 0:
                            copy_sky_line_list.append(new_add_sky_line)
                    # 对天际线列表重新进行排序
                    copy_sky_line_list = self.sort_sky_line_list_by_y_x(copy_sky_line_list)
                    # 遍历新的天际线列表，将可以合并的天际线进行合并
                    copy_sky_line_list = self.combine_sky_line(copy_sky_line_list)
                    # 计算分数
                    score = old_sky_line_cnt - len(copy_sky_line_list)
                    # 模块中心点坐标
                    center_position = self.calc_center_position(rotate_item, left_bottom_point)
                    # 生成当前得分对象
                    return Score(score, self.calc_link_score(center_position, rotate_item, result_list), 0, 0,
                                 copy_sky_line_list,
                                 Result(Item.copy(item), rotate_item, rotate_item.orient, center_position,
                                        sky_line, left_bottom_point))
        return None

    # 计算L块的得分
    def calc_L_score(self, item, rotate_item, result_list, sky_line_list):
        # 复制天际线列表
        copy_sky_line_list = sky_line_list.copy()
        # 记录原来天际线的个数
        old_sky_line_cnt = len(copy_sky_line_list)
        # 从天际线列表中取出第一个元素作为放置的天际线
        sky_line = copy_sky_line_list.pop(0)
        # ======================================================== 0/270度，往左放 并且 往右放 ========================================================
        if (rotate_item.type == '0' or rotate_item.type == '270') and rotate_item.w <= sky_line[2]:
            left_score = self.place_left_for_L_T(sky_line, item, rotate_item, copy_sky_line_list.copy(),
                                                 old_sky_line_cnt,
                                                 rotate_item.orient, result_list)
            right_score = self.place_right_for_L_T(sky_line, item, rotate_item, copy_sky_line_list.copy(),
                                                   old_sky_line_cnt,
                                                   rotate_item.orient, result_list)
            if left_score is None:
                return right_score
            if right_score is None:
                return left_score
            return left_score if left_score.deta_value > right_score.deta_value else right_score
        elif rotate_item.type == '90' and rotate_item.floor_horizontal_line[2] <= sky_line[2]:
            # ======================================================== 90度，往右放 ========================================================
            left_bottom_point = (sky_line[0] + sky_line[2] - rotate_item.w1, sky_line[1])
            # 如果超出边界，则直接返回None
            if self.is_out_of_bound(rotate_item, self.area,
                                    left_bottom_point):
                return None
            # 存在重叠，则直接返回None
            # 说明对标点不在包络矩形的左下角，所以是特殊的L型，所以在放置时，需要对其进行重叠判断，如果重叠则不能在这个点进行放置
            if self.is_overlap(result_list, rotate_item, left_bottom_point):
                return None
            # 到这里说明可以放
            # 对天际线列表重新进行排序
            copy_sky_line_list = self.sort_sky_line_list_by_x_y(copy_sky_line_list)
            # 找到右边的天际线进行切割，并计算浪费
            waste = 0
            # 找到第二高度
            b = False
            for i in range(len(copy_sky_line_list)):
                cs = copy_sky_line_list[i]
                if cs[0] == sky_line[0] + sky_line[2] and cs[1] >= sky_line[1]:
                    if cs[1] <= rotate_item.h1 + sky_line[1]:
                        # 上方浪费
                        if cs[2] < rotate_item.w - rotate_item.floor_horizontal_line[2]:
                            return None
                        # 判断浪费的空间里是否可以放下矩形,如果可以放下，则直接返回None
                        if self.waste_space_judge_place_rect_able is True and self.judge_waste_space_can_place_rect_able(
                                rotate_item.w - rotate_item.floor_horizontal_line[2],
                                rotate_item.h1 + sky_line[1] - cs[1]):
                            return None
                        copy_sky_line_list[i] = (cs[0] + (rotate_item.w - rotate_item.floor_horizontal_line[2]), cs[1],
                                                 cs[2] - (rotate_item.w - rotate_item.floor_horizontal_line[2]))
                        waste = (rotate_item.w - rotate_item.floor_horizontal_line[2]) * (
                                rotate_item.h1 + sky_line[1] - cs[1])
                    else:
                        # 下方浪费
                        # 判断浪费的空间里是否可以放下矩形,如果可以放下，则直接返回None
                        if self.waste_space_judge_place_rect_able is True and self.judge_waste_space_can_place_rect_able(
                                rotate_item.w - rotate_item.floor_horizontal_line[2], rotate_item.h1 + sky_line[1]):
                            return None
                        waste = (rotate_item.h1 + sky_line[1]) * (rotate_item.w - rotate_item.floor_horizontal_line[2])
                    b = True
                    if copy_sky_line_list[i][2] <= 0:
                        copy_sky_line_list.pop(i)
                    break
            if b is False:
                return None
            # 常规切割
            if sky_line[2] - rotate_item.floor_horizontal_line[2] > 0:
                copy_sky_line_list.append((sky_line[0], sky_line[1],
                                           sky_line[2] - rotate_item.floor_horizontal_line[2]))
            # 加入item潜在的天际线
            for add_sky_line in rotate_item.add_sky_line_list:
                new_add_sky_line = (
                    add_sky_line[0] + sky_line[0] + (sky_line[2] - rotate_item.floor_horizontal_line[2]),
                    add_sky_line[1] + sky_line[1], add_sky_line[2])
                for sk in copy_sky_line_list:
                    if sk[1] > new_add_sky_line[1] and str(sk) not in self.init_sky_line_set:
                        if sk[0] <= new_add_sky_line[0] and sk[0] + sk[2] >= new_add_sky_line[0]:
                            # 左边切
                            new_add_sky_line = (sk[0] + sk[2], new_add_sky_line[1],
                                                new_add_sky_line[0] + new_add_sky_line[2] - (sk[0] + sk[2]))
                            break
                        elif sk[0] >= new_add_sky_line[0] and sk[0] <= new_add_sky_line[0] + new_add_sky_line[2]:
                            # 右边切
                            new_add_sky_line = (new_add_sky_line[0], new_add_sky_line[1], sk[0] - new_add_sky_line[0])
                            break
                if new_add_sky_line[2] > 0:
                    copy_sky_line_list.append(new_add_sky_line)
            # 对天际线列表重新进行排序
            copy_sky_line_list = self.sort_sky_line_list_by_y_x(copy_sky_line_list)
            # 遍历新的天际线列表，将可以合并的天际线进行合并
            copy_sky_line_list = self.combine_sky_line(copy_sky_line_list)
            # 计算分数
            score = old_sky_line_cnt - len(copy_sky_line_list)
            # 模块中心点坐标
            center_position = self.calc_center_position(rotate_item, left_bottom_point)
            # 生成当前得分对象
            # return None
            return Score(score, self.calc_link_score(center_position, rotate_item, result_list), waste, 0,
                         copy_sky_line_list,
                         Result(Item.copy(item), rotate_item, rotate_item.orient, center_position,
                                sky_line, left_bottom_point))
        elif rotate_item.type == '180' and rotate_item.floor_horizontal_line[2] <= sky_line[2]:
            # ======================================================== 180度，往左放 ========================================================
            left_bottom_point = (sky_line[0] - rotate_item.w1, sky_line[1])
            # 如果超出边界，则直接返回None
            if self.is_out_of_bound(rotate_item, self.area,
                                    left_bottom_point):
                return None
            # 存在重叠，则直接返回None
            # 说明对标点不在包络矩形的左下角，所以是特殊的L型，所以在放置时，需要对其进行重叠判断，如果重叠则不能在这个点进行放置
            if self.is_overlap(result_list, rotate_item, left_bottom_point):
                return None
            # 到这里说明可以放
            # 对天际线列表重新进行排序
            copy_sky_line_list = self.sort_sky_line_list_by_x_y(copy_sky_line_list)
            # 找到左边的天际线进行切割，并计算浪费
            waste = 0
            # 找到第二高度
            b = False
            for i in range(len(copy_sky_line_list)):
                cs = copy_sky_line_list[i]
                if cs[0] + cs[2] == sky_line[0] and cs[1] >= sky_line[1]:
                    if cs[1] <= rotate_item.h2 + sky_line[1]:
                        # 上方浪费
                        if cs[2] < rotate_item.w - rotate_item.floor_horizontal_line[2]:
                            return None
                        # 判断浪费的空间里是否可以放下矩形,如果可以放下，则直接返回None
                        if self.waste_space_judge_place_rect_able is True and self.judge_waste_space_can_place_rect_able(
                                rotate_item.w - rotate_item.floor_horizontal_line[2],
                                rotate_item.h2 + sky_line[1] - cs[1]):
                            return None
                        copy_sky_line_list[i] = (
                            cs[0], cs[1], cs[2] - (rotate_item.w - rotate_item.floor_horizontal_line[2]))
                        waste = (rotate_item.w - rotate_item.floor_horizontal_line[2]) * (
                                rotate_item.h2 + sky_line[1] - cs[1])
                    else:
                        # 下方浪费
                        # 判断浪费的空间里是否可以放下矩形,如果可以放下，则直接返回None
                        if self.waste_space_judge_place_rect_able is True and self.judge_waste_space_can_place_rect_able(
                                rotate_item.w - rotate_item.floor_horizontal_line[2], rotate_item.h2 + sky_line[1]):
                            return None
                        waste = (rotate_item.h2 + sky_line[1]) * (rotate_item.w - rotate_item.floor_horizontal_line[2])
                    b = True
                    if copy_sky_line_list[i][2] <= 0:
                        copy_sky_line_list.pop(i)
                    break
            if b is False:
                return None
            # 常规切割
            if sky_line[2] - rotate_item.floor_horizontal_line[2] > 0:
                copy_sky_line_list.append((sky_line[0] + rotate_item.floor_horizontal_line[2], sky_line[1],
                                           sky_line[2] - rotate_item.floor_horizontal_line[2]))
            # 加入item潜在的天际线
            for add_sky_line in rotate_item.add_sky_line_list:
                new_add_sky_line = (
                    add_sky_line[0] + sky_line[0] - (rotate_item.w - rotate_item.floor_horizontal_line[2]),
                    add_sky_line[1] + sky_line[1], add_sky_line[2])
                for sk in copy_sky_line_list:
                    if sk[1] > new_add_sky_line[1] and str(sk) not in self.init_sky_line_set:
                        if sk[0] <= new_add_sky_line[0] and sk[0] + sk[2] >= new_add_sky_line[0]:
                            # 左边切
                            new_add_sky_line = (sk[0] + sk[2], new_add_sky_line[1],
                                                new_add_sky_line[0] + new_add_sky_line[2] - (sk[0] + sk[2]))
                            break
                        elif sk[0] >= new_add_sky_line[0] and sk[0] <= new_add_sky_line[0] + new_add_sky_line[2]:
                            # 右边切
                            new_add_sky_line = (new_add_sky_line[0], new_add_sky_line[1], sk[0] - new_add_sky_line[0])
                            break
                if new_add_sky_line[2] > 0:
                    copy_sky_line_list.append(new_add_sky_line)
            # 对天际线列表重新进行排序
            copy_sky_line_list = self.sort_sky_line_list_by_y_x(copy_sky_line_list)
            # 遍历新的天际线列表，将可以合并的天际线进行合并
            copy_sky_line_list = self.combine_sky_line(copy_sky_line_list)
            # 计算分数
            score = old_sky_line_cnt - len(copy_sky_line_list)
            # 模块中心点坐标
            center_position = self.calc_center_position(rotate_item, left_bottom_point)
            # 生成当前得分对象
            # return None
            return Score(score, self.calc_link_score(center_position, rotate_item, result_list), waste, 0,
                         copy_sky_line_list,
                         Result(Item.copy(item), rotate_item, rotate_item.orient, center_position,
                                sky_line, left_bottom_point))
        return None

    # 计算矩形块的得分
    def calc_S_score(self, item, rotate_item, result_list, sky_line_list):
        # 复制一份天际线列表
        copy_sky_line_list = sky_line_list.copy()
        # 从天际线列表中取出第一个元素作为放置的天际线
        sky_line = copy_sky_line_list.pop(0)
        # 矩形的评分
        score = 0
        # 找出当前天际线左右两边的墙高度 获取 (左hl，右hr)
        hl, hr = self.hl_hr
        # 根据八种情况得分
        if hl >= hr:
            # 左边墙比右边墙高
            if rotate_item.w == sky_line[2] and rotate_item.h == hl:
                score = 7
            elif rotate_item.w == sky_line[2] and rotate_item.h == hr:
                score = 6
            elif rotate_item.w == sky_line[2] and rotate_item.h > hl:
                score = 5
            elif rotate_item.w < sky_line[2] and rotate_item.h == hl:
                score = 4
            elif rotate_item.w == sky_line[2] and rotate_item.h < hl and rotate_item.h > hr:
                score = 3
            elif rotate_item.w < sky_line[2] and rotate_item.h == hr:
                score = 2  # 靠右
            elif rotate_item.w == sky_line[2] and rotate_item.h < hr:
                score = 1
            elif rotate_item.w < sky_line[2] and rotate_item.h != hl:
                score = 0
            else:
                return None
            if score == 2:
                # 靠右
                # 左下角点
                left_bottom_point = (sky_line[0] + sky_line[2] - rotate_item.w, sky_line[1])
                # 判断矩形是否超出边界
                if self.is_out_of_bound(rotate_item, self.area, left_bottom_point):
                    return None
                # 将矩形顶部的天际线加入
                for add_sky_line in rotate_item.add_sky_line_list:
                    copy_sky_line_list.append(
                        (add_sky_line[0] + sky_line[0] + sky_line[2] - rotate_item.w, add_sky_line[1] + sky_line[1],
                         add_sky_line[2]))
                # 常规切割
                if sky_line[2] - rotate_item.w > 0:
                    copy_sky_line_list.append((sky_line[0], sky_line[1],
                                               sky_line[2] - rotate_item.w))
            else:
                # 靠左
                # 左下角点
                left_bottom_point = (sky_line[0], sky_line[1])
                # 判断矩形是否超出边界
                if self.is_out_of_bound(rotate_item, self.area, left_bottom_point):
                    return None
                # 将矩形顶部的天际线加入
                for add_sky_line in rotate_item.add_sky_line_list:
                    copy_sky_line_list.append(
                        (add_sky_line[0] + sky_line[0], add_sky_line[1] + sky_line[1], add_sky_line[2]))
                # 常规切割
                if sky_line[2] - rotate_item.w > 0:
                    copy_sky_line_list.append((sky_line[0] + rotate_item.w, sky_line[1],
                                               sky_line[2] - rotate_item.w))
            # 对天际线列表重新进行排序
            copy_sky_line_list = self.sort_sky_line_list_by_y_x(copy_sky_line_list)
            # 遍历新的天际线列表，将可以合并的天际线进行合并
            copy_sky_line_list = self.combine_sky_line(copy_sky_line_list)
            # 模块中心点坐标
            center_position = self.calc_center_position(rotate_item, left_bottom_point)
            return Score(0, self.calc_link_score(center_position, rotate_item, result_list), 0, score,
                         copy_sky_line_list,
                         Result(Item.copy(item), rotate_item, rotate_item.orient, center_position,
                                sky_line,
                                left_bottom_point))
        else:
            # 右边墙比左边墙高
            if rotate_item.w == sky_line[2] and rotate_item.h == hr:
                score = 7
            elif rotate_item.w == sky_line[2] and rotate_item.h == hl:
                score = 6
            elif rotate_item.w == sky_line[2] and rotate_item.h > hr:
                score = 5
            elif rotate_item.w < sky_line[2] and rotate_item.h == hr:
                score = 4  # 靠右
            elif rotate_item.w == sky_line[2] and rotate_item.h < hr and rotate_item.h > hl:
                score = 3
            elif rotate_item.w < sky_line[2] and rotate_item.h == hl:
                score = 2
            elif rotate_item.w == sky_line[2] and rotate_item.h < hl:
                score = 1
            elif rotate_item.w < sky_line[2] and rotate_item.h != hr:
                score = 0  # 靠右
            else:
                return None
            if score == 4 or score == 0:
                # 靠右
                # 左下角点
                left_bottom_point = (sky_line[0] + sky_line[2] - rotate_item.w, sky_line[1])
                # 判断矩形是否超出边界
                if self.is_out_of_bound(rotate_item, self.area, left_bottom_point):
                    return None
                # 将矩形顶部的天际线加入
                for add_sky_line in rotate_item.add_sky_line_list:
                    copy_sky_line_list.append(
                        (add_sky_line[0] + sky_line[0] + sky_line[2] - rotate_item.w, add_sky_line[1] + sky_line[1],
                         add_sky_line[2]))
                # 常规切割
                if sky_line[2] - rotate_item.w > 0:
                    copy_sky_line_list.append((sky_line[0], sky_line[1],
                                               sky_line[2] - rotate_item.w))
            else:
                # 靠左
                # 左下角点
                left_bottom_point = (sky_line[0], sky_line[1])
                # 判断矩形是否超出边界
                if self.is_out_of_bound(rotate_item, self.area, left_bottom_point):
                    return None
                # 将矩形顶部的天际线加入
                for add_sky_line in rotate_item.add_sky_line_list:
                    copy_sky_line_list.append(
                        (add_sky_line[0] + sky_line[0], add_sky_line[1] + sky_line[1], add_sky_line[2]))
                # 常规切割
                if sky_line[2] - rotate_item.w > 0:
                    copy_sky_line_list.append((sky_line[0] + rotate_item.w, sky_line[1],
                                               sky_line[2] - rotate_item.w))
            # 对天际线列表重新进行排序
            copy_sky_line_list = self.sort_sky_line_list_by_y_x(copy_sky_line_list)
            # 遍历新的天际线列表，将可以合并的天际线进行合并
            copy_sky_line_list = self.combine_sky_line(copy_sky_line_list)
            # 模块中心点坐标
            center_position = self.calc_center_position(rotate_item, left_bottom_point)
            return Score(0, self.calc_link_score(center_position, rotate_item, result_list), 0, score,
                         copy_sky_line_list,
                         Result(Item.copy(item), rotate_item, rotate_item.orient, center_position,
                                sky_line,
                                left_bottom_point))

    # 判断浪费的空间里是否可以放下矩形
    def judge_waste_space_can_place_rect_able(self, waste_w, waste_h):
        for i in range(len(self.item_flags)):
            if self.item_flags[i] == 0:
                item = self.items[i]
                if (item.w <= waste_w and item.h <= waste_h) or (item.w <= waste_h and item.h <= waste_w):
                    return True
        return False

    # 用于L/T型的往左放
    def place_left_for_L_T(self, sky_line, item, rotate_item, copy_sky_line_list, old_sky_line_cnt, orient,
                           result_list):
        # 往左放
        left_bottom_point = (sky_line[0], sky_line[1])
        # 如果超出边界，则直接返回None
        if self.is_out_of_bound(rotate_item, self.area,
                                left_bottom_point):
            return None
        # 加入item潜在的天际线
        for add_sky_line in rotate_item.add_sky_line_list:
            copy_sky_line_list.append(
                (add_sky_line[0] + sky_line[0], add_sky_line[1] + sky_line[1], add_sky_line[2]))
        # 常规切割
        if sky_line[2] - rotate_item.w > 0:
            copy_sky_line_list.append((sky_line[0] + rotate_item.w, sky_line[1],
                                       sky_line[2] - rotate_item.w))
        # 对天际线列表重新进行排序
        copy_sky_line_list = self.sort_sky_line_list_by_y_x(copy_sky_line_list)
        # 遍历新的天际线列表，将可以合并的天际线进行合并
        copy_sky_line_list = self.combine_sky_line(copy_sky_line_list)
        # 计算分数
        score = old_sky_line_cnt - len(copy_sky_line_list)
        # 模块中心点坐标
        center_position = self.calc_center_position(rotate_item, left_bottom_point)
        # 生成当前得分对象
        return Score(score, self.calc_link_score(center_position, rotate_item, result_list), 0, 0, copy_sky_line_list,
                     Result(Item.copy(item), rotate_item, orient, center_position,
                            sky_line,
                            left_bottom_point))

    # 用于L/T型的往右放
    def place_right_for_L_T(self, sky_line, item, rotate_item, copy_sky_line_list, old_sky_line_cnt, orient,
                            result_list):
        # 往右边放
        left_bottom_point = (sky_line[0] + sky_line[2] - rotate_item.floor_horizontal_line[2], sky_line[1])
        # 如果超出边界，则直接返回None
        if self.is_out_of_bound(rotate_item, self.area,
                                left_bottom_point):
            return None
        # 加入item潜在的天际线
        for add_sky_line in rotate_item.add_sky_line_list:
            copy_sky_line_list.append(
                (add_sky_line[0] + sky_line[0] + (sky_line[2] - rotate_item.floor_horizontal_line[2]),
                 add_sky_line[1] + sky_line[1], add_sky_line[2]))
        # 常规切割
        if sky_line[2] - rotate_item.w > 0:
            copy_sky_line_list.append((sky_line[0], sky_line[1],
                                       sky_line[2] - rotate_item.w))
        # 对天际线列表重新进行排序
        copy_sky_line_list = self.sort_sky_line_list_by_y_x(copy_sky_line_list)
        # 遍历新的天际线列表，将可以合并的天际线进行合并
        copy_sky_line_list = self.combine_sky_line(copy_sky_line_list)
        # 计算分数
        score = old_sky_line_cnt - len(copy_sky_line_list)
        # 模块中心点坐标
        center_position = self.calc_center_position(rotate_item, left_bottom_point)
        # 生成当前得分对象
        return Score(score, self.calc_link_score(center_position, rotate_item, result_list), 0, 0, copy_sky_line_list,
                     Result(Item.copy(item), rotate_item, orient, center_position,
                            sky_line,
                            left_bottom_point))

    # 找出当前天际线左右两边的墙高度 获取 (较高的墙h1，较矮的墙h2)
    def get_h1_h2_by_sky_line(self, sky_line, sky_line_list):
        sky_line_1 = None  # 左边的墙
        sky_line_2 = None  # 右边的墙
        for sk in sky_line_list:
            if sk[1] > sky_line[1]:
                if sk[0] + sk[2] == sky_line[0] and (sky_line_1 is None or sky_line_1[1] > sk[1]):
                    # 左边的墙
                    sky_line_1 = sk
                elif sky_line[0] + sky_line[2] == sk[0] and (sky_line_2 is None or sky_line_2[1] > sk[1]):
                    # 右边的墙
                    sky_line_2 = sk
        h1 = sky_line_1[1] if sky_line_1 is not None else 0
        h2 = sky_line_2[1] if sky_line_2 is not None else 0
        return h1, h2

    # 判断放在该天际线上，是否会超出天际线长度
    def is_out_of_sky_line(self, item, sky_line):
        # 其实就是查看其包络矩形宽度有没有超出天际线长度
        return item.w > sky_line[2]

    # 判断放在该天际线上，是否会超出边界
    def is_out_of_bound(self, item, area, left_bottom_point):
        # 其实就是判断当前模块是否被边界包含
        # 如果当前模块被多边形边界包含，则没有超出边界
        b = True
        # item_bound = Polygon(self.get_absolute_boundary_by_sky_line(item.boundary.copy(), left_bottom_point))
        # global_bound = Polygon(area)
        # b = global_bound.contains(item_bound)

        if len(self.area) == 4:
            for sk in item.add_sky_line_list:
                # 天际线左端点
                c = 0
                p = (sk[0], sk[1])
                if c == 0:
                    for horizontal_line in self.area_horizontal_line_list:
                        if horizontal_line[1] >= p[1] + left_bottom_point[1] and horizontal_line[0] <= p[0] + \
                                left_bottom_point[0] <= horizontal_line[0] + horizontal_line[2]:
                            c += 1
                    if c % 2 == 0:
                        b = False
                        break
                # 天际线右端点
                c = 0
                p = (sk[0] + sk[2], sk[1])
                if c == 0:
                    for horizontal_line in self.area_horizontal_line_list:
                        if horizontal_line[1] >= p[1] + left_bottom_point[1] and horizontal_line[0] <= p[0] + \
                                left_bottom_point[0] <= horizontal_line[0] + horizontal_line[2]:
                            c += 1
                    if c % 2 == 0:
                        b = False
                        break
        else:
            if len(item.boundary) == 6:
                if item.type == '0':
                    b = self.judge_rect_out_of_bound(item.w1, item.h, left_bottom_point) is False
                    if b is True:
                        b = self.judge_rect_out_of_bound(item.w2, item.h2, (
                            left_bottom_point[0] + item.w1, left_bottom_point[1])) is False
                elif item.type == '90':
                    b = self.judge_rect_out_of_bound(item.w1, item.h, left_bottom_point) is False
                    if b is True:
                        b = self.judge_rect_out_of_bound(item.w2, item.h2, (
                            left_bottom_point[0] + item.w1, left_bottom_point[1] + item.h1)) is False
                elif item.type == '180':
                    b = self.judge_rect_out_of_bound(item.w1, item.h1,
                                                     (left_bottom_point[0], left_bottom_point[1] + item.h2)) is False
                    if b is True:
                        b = self.judge_rect_out_of_bound(item.w2, item.h, (
                            left_bottom_point[0] + item.w1, left_bottom_point[1])) is False
                elif item.type == '270':
                    b = self.judge_rect_out_of_bound(item.w1, item.h1, left_bottom_point) is False
                    if b is True:
                        b = self.judge_rect_out_of_bound(item.w2, item.h, (
                            left_bottom_point[0] + item.w1, left_bottom_point[1])) is False
                else:
                    raise RuntimeError("不应该传进来的类型：", item.type)
            elif len(item.boundary) == 8:
                if item.type == '0':
                    # 竖着切
                    b = self.judge_rect_out_of_bound(item.w1, item.h1, left_bottom_point) is False
                    if b is True:
                        b = self.judge_rect_out_of_bound(item.w2, item.h, (
                            left_bottom_point[0] + item.w1, left_bottom_point[1])) is False
                    if b is True:
                        b = self.judge_rect_out_of_bound(item.w3, item.h4, (
                            left_bottom_point[0] + item.w1 + item.w2, left_bottom_point[1])) is False
                elif item.type == '90':
                    # 横切
                    b = self.judge_rect_out_of_bound(item.w1, item.h3, left_bottom_point) is False
                    if b is True:
                        b = self.judge_rect_out_of_bound(item.w, item.h2, (
                            left_bottom_point[0], left_bottom_point[1] + item.h3)) is False
                    if b is True:
                        b = self.judge_rect_out_of_bound(item.w4, item.h1, (
                            left_bottom_point[0], left_bottom_point[1] + item.h2 + item.h3)) is False
                elif item.type == '180':
                    # 竖切
                    b = self.judge_rect_out_of_bound(item.w1, item.h1,
                                                     (left_bottom_point[0], left_bottom_point[1] + item.h2)) is False
                    if b is True:
                        b = self.judge_rect_out_of_bound(item.w2, item.h, (
                            left_bottom_point[0] + item.w1, left_bottom_point[1])) is False
                    if b is True:
                        b = self.judge_rect_out_of_bound(item.w3, item.h4, (
                            left_bottom_point[0] + item.w1 + item.w2, left_bottom_point[1] + item.h3)) is False
                elif item.type == '270':
                    # 横切
                    b = self.judge_rect_out_of_bound(item.w1, item.h1,
                                                     (left_bottom_point[0] + item.w2, left_bottom_point[1])) is False
                    if b is True:
                        b = self.judge_rect_out_of_bound(item.w, item.h2, (
                            left_bottom_point[0], left_bottom_point[1] + item.h1)) is False
                    if b is True:
                        b = self.judge_rect_out_of_bound(item.w4, item.h3, (
                            left_bottom_point[0] + item.w3, left_bottom_point[1] + item.h1 + item.h2)) is False
                else:
                    raise RuntimeError("不应该传进来的类型：", item.type)
            else:
                b = self.judge_rect_out_of_bound(item.w, item.h, left_bottom_point) is False
        return b is False

    # 判断一个矩形在不在边界内
    def judge_rect_out_of_bound(self, w, h, left_bottom_point):
        b = True
        for horizontal_line in self.area_horizontal_line_list:
            if left_bottom_point[1] + h > horizontal_line[1] > left_bottom_point[
                1] and WskhFunction.line_is_intersect((horizontal_line[0], horizontal_line[0] + horizontal_line[2]),
                                                      (left_bottom_point[0], left_bottom_point[0] + w)) is True:
                b = False
                break
        if b is True:
            for vertical_line in self.area_vertical_line_list:
                if left_bottom_point[0] + w > vertical_line[0] > left_bottom_point[
                    0] and WskhFunction.line_is_intersect((vertical_line[1], vertical_line[1] + vertical_line[2]),
                                                          (left_bottom_point[1], left_bottom_point[1] + h)):
                    b = False
                    break
        if b is True:
            top_line = None
            bottom_line = None
            for horizontal_line in self.area_horizontal_line_list:
                if WskhFunction.line_is_intersect((horizontal_line[0], horizontal_line[0] + horizontal_line[2]),
                                                  (left_bottom_point[0], left_bottom_point[0] + w)) is True:
                    if horizontal_line[1] >= left_bottom_point[1] + h:
                        if top_line is None or top_line[1] > horizontal_line[1]:
                            top_line = horizontal_line
                    elif horizontal_line[1] <= left_bottom_point[1]:
                        if bottom_line is None or bottom_line[1] < horizontal_line[1]:
                            bottom_line = horizontal_line
            if top_line is None or bottom_line is None:
                b = False
            else:
                if str(bottom_line) not in self.init_sky_line_set or str(top_line) in self.init_sky_line_set:
                    b = False
        return b is False

    # 判断一个水平线段是否为初始化的天际线
    def judge_horizontal_line_is_init_skyline(self, horizontal_line):
        return str(horizontal_line) in self.init_sky_line_set

    # 根据模块和当前放置的天际线，获取模块放置的中心点坐标（包络矩形中心点）
    def calc_center_position(self, item, left_bottom_point):
        return left_bottom_point[0] + item.w / 2.0, left_bottom_point[1] + item.h / 2.0

    # 初始化天际线列表
    def init_sky_line_list(self, area):
        init_sky_line_list = WskhFunction.calc_horizontal_line_list(area)
        lst = []
        self.init_sky_line_set = set()
        for isk1 in init_sky_line_list:
            c = 0
            for isk2 in init_sky_line_list:
                if isk2[0] + isk2[2] > isk1[0] >= isk2[0] and isk2[1] > isk1[1]:
                    if WskhFunction.line_is_intersect((isk1[0], isk1[0] + isk1[2]),
                                                                            (isk2[0], isk2[0] + isk2[2])):
                        c += 1
            if c % 2 != 0:
                self.init_sky_line_set.add(str(isk1))
                lst.append(isk1)
        return lst

    # 当前天际线无法放置任何一个模块，故对其进行上移合并处理
    def up_move_sky_line(self, sky_line_list):
        # 用sky_line_list的第一个元素，对其进行上移合并
        sky_line_0 = sky_line_list[0]
        # 初始化要和sky_line_0进行合并的对象sky_line_1(右)和sky_line_2(左)
        sky_line_1 = None
        sky_line_1_index = -1
        sky_line_2 = None
        sky_line_2_index = -1
        for i in range(len(sky_line_list)):
            if i != 0:
                sky_line_i = sky_line_list[i]
                if sky_line_i[1] >= sky_line_0[1]:
                    if sky_line_0[0] + sky_line_0[2] == sky_line_i[0] and (
                            sky_line_1 is None or sky_line_1[1] > sky_line_i[1]):
                        sky_line_1 = sky_line_i
                        sky_line_1_index = i
                    elif sky_line_0[0] == sky_line_i[0] + sky_line_i[2] and (
                            sky_line_2 is None or sky_line_2[1] > sky_line_i[1]):
                        sky_line_2 = sky_line_i
                        sky_line_2_index = i
        # 如果sky_line_1和sky_line_2都是None，说明不能合并，直接返回
        if sky_line_1 is None and sky_line_2 is None:
            return sky_line_list
        # 如果左右两边不等高,则选取较矮的合并
        if sky_line_1 is not None and sky_line_2 is not None:
            if sky_line_1[1] > sky_line_2[1]:
                sky_line_1 = None
            elif sky_line_1[1] < sky_line_2[1]:
                sky_line_2 = None
        # 否则，对sky_line_0和sky_line_1进行合并
        if sky_line_2 is None:
            # 和右边的合并
            sky_line_0 = sky_line_list.pop(0)
            sky_line_1 = sky_line_list.pop(sky_line_1_index - 1)
            sky_line_combine = (sky_line_0[0], sky_line_1[1], sky_line_0[2] + sky_line_1[2])
            sky_line_combine = self.correct_sky_line_by_area_vertical_line_list(sky_line_combine)
            if sky_line_combine[2] > 0:
                sky_line_list.append(sky_line_combine)
        elif sky_line_1 is None:
            # 和左边的合并
            sky_line_0 = sky_line_list.pop(0)
            sky_line_2 = sky_line_list.pop(sky_line_2_index - 1)
            sky_line_combine = (sky_line_2[0], sky_line_2[1], sky_line_0[2] + sky_line_2[2])
            sky_line_combine = self.correct_sky_line_by_area_vertical_line_list(sky_line_combine)
            if sky_line_combine[2] > 0:
                sky_line_list.append(sky_line_combine)
        else:
            # 两边都要合并
            sky_line_0 = sky_line_list.pop(0)
            if sky_line_1_index > sky_line_2_index:
                sky_line_1 = sky_line_list.pop(sky_line_1_index - 1)
                sky_line_2 = sky_line_list.pop(sky_line_2_index - 1)
            else:
                sky_line_2 = sky_line_list.pop(sky_line_2_index - 1)
                sky_line_1 = sky_line_list.pop(sky_line_1_index - 1)
            sky_line_combine = (sky_line_2[0], sky_line_2[1], sky_line_0[2] + sky_line_1[2] + sky_line_2[2])
            sky_line_combine = self.correct_sky_line_by_area_vertical_line_list(sky_line_combine)
            if sky_line_combine[2] > 0:
                sky_line_list.append(sky_line_combine)
        return sky_line_list

    # 传入一个天际线，根据边界的垂直线段列表，对其进行修正后返回
    def correct_sky_line_by_area_vertical_line_list(self, sky_line):
        min_dis = None
        left_v_line = None
        for v_line in self.area_vertical_line_list:
            if v_line[1] <= sky_line[1] <= v_line[1] + v_line[2]:
                if v_line[0] > sky_line[0]:
                    d = v_line[0] - sky_line[0]
                    if min_dis is None or min_dis > d:
                        min_dis = d
                if v_line[0] <= sky_line[0]:
                    if left_v_line is None or abs(left_v_line[0] - sky_line[0]) > abs(v_line[0] - sky_line[0]):
                        left_v_line = v_line
        if left_v_line is None:
            sky_line = (sky_line[0] + min_dis, sky_line[1], sky_line[2] - min_dis)
        elif min_dis is not None:
            sky_line = (sky_line[0], sky_line[1], min(sky_line[2], min_dis))
        return sky_line

    # 遍历新的天际线列表(需要是排序后的)，将可以合并的天际线进行合并
    def combine_sky_line(self, sky_line_list):
        i = 0
        while i + 1 < len((sky_line_list)):
            j = i + 1
            sky_line_i = sky_line_list[i]
            sky_line_j = sky_line_list[j]
            # 如果天际线i的尾部正好等于天际线j的头部，那么天际线i和j就可以进行合并
            if sky_line_i[1] == sky_line_j[1] and sky_line_i[0] + sky_line_i[2] == sky_line_j[0]:
                sky_line_list.pop(min(i, j))
                sky_line_list.pop(max(i, j) - 1)
                sky_line_combine = (sky_line_i[0], sky_line_i[1], sky_line_i[2] + sky_line_j[2])
                # sky_line_combine = self.correct_sky_line_by_area_vertical_line_list(sky_line_combine)
                if sky_line_combine[2] > 0:
                    sky_line_list.append(sky_line_combine)
                continue
            i += 1
        return sky_line_list

    # 传入已放置模块和当前准备放置的模块及天际线，判断准备要放置的模块是否会造成重叠
    # def is_overlap(self, result_list, item, left_bottom_point):
    #     # 获取当前放置模块的多边形对象
    #     s = time.time()
    #     item_bound = Polygon(self.get_absolute_boundary_by_sky_line(item.boundary.copy(), left_bottom_point))
    #     for result in result_list:
    #         if result.left_bottom_point[0] + result.rotate_item.w <= left_bottom_point[0]:
    #             continue  # 在item左边
    #         elif result.left_bottom_point[0] >= left_bottom_point[0] + item.w:
    #             continue  # 在item右边
    #         elif result.left_bottom_point[1] + result.rotate_item.h <= left_bottom_point[1]:
    #             continue  # 在item下边
    #         elif result.left_bottom_point[1] >= result.rotate_item.h + left_bottom_point[1]:
    #             continue  # 在item上边
    #         # 如果他们的包络矩形重合，则他们有可能重合，进行多边形重合判断
    #         absolute_boundary = self.get_absolute_boundary_by_sky_line(result.rotate_item.boundary.copy(),
    #                                                                    result.left_bottom_point)
    #         if item_bound.overlaps(Polygon(absolute_boundary)):
    #             self.timer += (time.time() - s)
    #             return True
    #     self.timer += (time.time()-s)
    #     return False

    # TODO 自定义多边形重叠判断
    def is_overlap(self, result_list, item, left_bottom_point):
        # 获取当前放置模块的多边形对象
        for result in result_list:
            if result.left_bottom_point[0] + result.rotate_item.w <= left_bottom_point[0]:
                continue  # 在item左边
            elif result.left_bottom_point[0] >= left_bottom_point[0] + item.w:
                continue  # 在item右边
            elif result.left_bottom_point[1] + result.rotate_item.h <= left_bottom_point[1]:
                continue  # 在item下边
            elif result.left_bottom_point[1] >= result.rotate_item.h + left_bottom_point[1]:
                continue  # 在item上边
            # 如果他们的包络矩形重合，则他们有可能重合，进行多边形重合判断
            # if CzgFunction.is_overlap(item, left_bottom_point, result.rotate_item,
            #                           result.left_bottom_point) or CzgFunction.is_overlap(result.rotate_item,
            #                                                                               result.left_bottom_point,
            #                                                                               item, left_bottom_point):
            if WskhFunction.is_overlap(item, left_bottom_point, result.rotate_item, result.left_bottom_point) is True:
                return True
        return False

    # 计算link得分
    def calc_link_score(self, center_position, rotate_item, result_list):
        # 计算连接得分
        range_w, range_h = 300, 300
        link_value = 0
        for result in result_list:
            for point in result.rotate_item.boundary:
                if center_position[0] - range_w <= point[0] + result.left_bottom_point[0] <= center_position[
                    0] + range_w and center_position[
                    1] - range_h <= point[1] + result.left_bottom_point[1] <= center_position[1] + range_h:
                    for link in self.links:
                        if rotate_item.name in link.link_dict and result.rotate_item.name in link.link_dict:
                            port = result.rotate_item.ports[int(link.link_dict[result.rotate_item.name]) - 1]
                            for p in port:
                                if center_position[0] - range_w <= p[0] + result.left_bottom_point[0] <= \
                                        center_position[0] + range_w and center_position[
                                    1] - range_h <= p[1] + result.left_bottom_point[1] <= center_position[
                                    1] + range_h:
                                    link_value += 1
                                break
                    break
        return link_value

    # 传入相对包络矩形坐标的顶点列表和sky_line，返回绝对坐标的顶点列表
    def get_absolute_boundary_by_sky_line(self, boundary, left_bottom_point):
        for i in range(len(boundary)):
            boundary[i] = (boundary[i][0] + left_bottom_point[0], boundary[i][1] + left_bottom_point[1])
        return boundary

    # 对天际线列表进行y_x排序
    def sort_sky_line_list_by_y_x(self, sky_line_list):
        sky_line_list.sort(key=functools.cmp_to_key(my_compare_by_y_x))
        return sky_line_list

    # 对天际线列表进行x_y排序
    def sort_sky_line_list_by_x_y(self, sky_line_list):
        sky_line_list.sort(key=functools.cmp_to_key(my_compare_by_x_y))
        return sky_line_list


# 自定义排序规则
def my_compare_by_y_x(sky_line_1, sky_line_2):
    if sky_line_1[1] < sky_line_2[1]:
        return -1
    elif sky_line_1[1] > sky_line_2[1]:
        return 1
    elif sky_line_1[0] < sky_line_2[0]:
        return -1
    elif sky_line_1[0] > sky_line_2[0]:
        return 1
    elif sky_line_1[2] < sky_line_2[2]:
        return -1
    elif sky_line_1[2] > sky_line_2[2]:
        return 1
    else:
        return 0


def my_compare_by_x_y(sky_line_1, sky_line_2):
    if sky_line_1[0] < sky_line_2[0]:
        return -1
    elif sky_line_1[0] > sky_line_2[0]:
        return 1
    elif sky_line_1[1] < sky_line_2[1]:
        return -1
    elif sky_line_1[1] > sky_line_2[1]:
        return 1
    elif sky_line_1[2] < sky_line_2[2]:
        return -1
    elif sky_line_1[2] > sky_line_2[2]:
        return 1
    else:
        return 0


# 分数类
class Score:
    def __init__(self, deta_value, link_value, waste_value, rect_value, sky_line_list, result):
        self.deta_value = deta_value
        self.link_value = link_value
        self.waste_value = waste_value
        self.rect_value = rect_value
        self.sky_line_list = sky_line_list
        self.result = result

    def to_string(self):
        return 'Score', {
            'deta_value': self.deta_value,
            'link_value': self.link_value,
            'waste_value': self.waste_value,
            'rect_value': self.rect_value,
        }
