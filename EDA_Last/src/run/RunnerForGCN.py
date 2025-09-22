import os
import sys
# sys.path.append('/home/eda220806/project/code')
import argparse

from torch.cuda import seed_all

import src.function.CzgFunction2 as CzgFunction2
import src.function.ReadFileUtils as RedaFileUtils
from src.model.ClassificationAndRegression.HeuModel import *
import torch
import time
import functools
from src.function.RoutingScore_LAHC_V4 import RoutingScore_Interface, TimeoutException, DataUtils

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning, module='sklearn')

def set_global_seed(seed):
    """
    设置项目中所有主要随机源的种子，以确保实验的可复现性。

    Args:
        seed (int): 要设置的种子值。
    """
    # 1. 设置 Python 内置 random 模块的种子
    random.seed(seed)

    # 2. 设置 NumPy 的种子
    np.random.seed(seed)

    # 3. 尝试为 PyTorch 设置种子（如果已安装）
    try:
        import torch

        # 为所有 CPU 设置种子
        torch.manual_seed(seed)

        # 如果 CUDA 可用，为所有 GPU 设置种子
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 4. 配置 cuDNN 的行为
        # 这些设置可以确保 cuDNN 的卷积算法是确定性的，但可能会牺牲性能
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    except ImportError:
        # 如果项目中没有 PyTorch，则静默跳过
        pass

    # 5. 设置 PYTHONHASHSEED 环境变量
    # 注意：这行代码必须在 Python 解释器启动前执行才能生效。
    # 在脚本内部设置通常是为了记录和提醒。
    # 正确用法是在命令行中启动: `PYTHONHASHSEED=seed_value python your_script.py`
    os.environ['PYTHONHASHSEED'] = str(seed)


# 传入相对包络矩形坐标的顶点列表和left_bottom_point，返回绝对坐标的顶点列表
def get_absolute_position_by_left_bottom_point(boundary, left_bottom_point):
    for i in range(len(boundary)):
        boundary[i] = [boundary[i][0] + left_bottom_point[0], boundary[i][1] + left_bottom_point[1]]
    return boundary


def items_compare_by_area_decrease(item1, item2):
    if item1.area > item2.area:
        return -1
    elif item1.area < item2.area:
        return 1
    else:
        return 0


def items_compare_by_area_longest_line(item1, item2):
    if max(item1.w, item1.h) > max(item2.w, item2.h):
        return -1
    elif max(item1.w, item1.h) < max(item2.w, item2.h):
        return 1
    else:
        return 0


def items_compare_by_area_polygon_difficulty(item1, item2):
    l1 = len(item1.boundary)
    l2 = len(item2.boundary)
    if l1 > l2:
        return -1
    elif l1 < l2:
        return 1
    else:
        return 0

if __name__ == '__main__':
    set_global_seed(520)


    # 初始化参数解析器
    parser = argparse.ArgumentParser(description="处理 EDA 数据样本。")
    parser.add_argument("--start", type=int, default=1, help="处理的起始索引。")
    parser.add_argument("--end", type=int, default=3, help="处理的结束索引 (不包含此索引)。")

    args = parser.parse_args()

    # 使用解析后的参数
    start = args.start
    end = args.end

    # --- 动态获取项目根目录 ---
    # __file__ 是当前脚本 (Runner01.py) 的绝对路径
    # 例如: D:\IDEA2020\Project\EDA_Last\src\run\Runner01.py
    current_script_path = os.path.abspath(__file__)
    # run_dir 是脚本所在的目录: D:\IDEA2020\Project\EDA_Last\src\run
    run_dir = os.path.dirname(current_script_path)
    # src_root_dir 是 src 目录: D:\IDEA2020\Project\EDA_Last\src
    src_root_dir = os.path.dirname(run_dir)
    # project_root 是项目的根目录: D:\IDEA2020\Project\EDA_Last
    project_root = os.path.dirname(src_root_dir)

    #Todo: 记得改
    # sample5_45 = [5,10,16,20,25,30,40,45]
    sample5_45 = [45]
    GCN_ON = True
    Classify_ON = False
    id = f"{start}_{end}" # 跑每个案例的 1 到 60 然后分开机器跑

    for sample in sample5_45:
        base_path = os.path.join(project_root, 'src', 'data', 'EDA_DATA', 'sample' + str(sample))

        # 原始: ports_link_input_txt_path = r'D:\IDEA2020\Project\EDA_Last\src\data\EDA_DATA\connect\connect_file\connect_'+ str(sample) +'.txt'
        ports_link_input_txt_path = os.path.join(project_root, 'src', 'data', 'EDA_DATA', 'connect', 'connect_file',
                                                 'connect_' + str(sample) + '.txt')

        # --- 输出文件路径 (相对于项目根目录) ---
        output_path = os.path.join(project_root, 'src', 'result', 'GCN', 'result_'+ id + '.csv')
        # output_path = os.path.join(project_root, 'src', 'result', 'GCN+Classify', 'result_' + id + '.csv')

        skip_path_logdir = os.path.join(project_root, 'src', 'result', 'GCN', 'skip_logs', 'sample' + str(sample))
        # skip_path_logdir = os.path.join(project_root, 'src', 'result', 'GCN+Classify', 'skip_logs', 'sample' + str(sample))

        for i in range(start, end):
            # for i in range(5, 6): #测试用
            folder_name = f"{sample}-{i}"
            ports_area_input_txt_path = os.path.join(base_path, folder_name, 'placement_info.txt')

            generate_layoutForGCNTrain_txt_output_path = f"D:/本科/比赛/2022.09.01 EDA图像拼接/2024更新布线算法/data/layoutForGCNTrain/{folder_name}/placement_info.txt"

            # 检查文件是否存在，如果不存在则跳过本次循环
            if not os.path.exists(ports_area_input_txt_path):
                RedaFileUtils.create_file_from_skip_message(folder_name, skip_path_logdir)
                print(f"文件不存在，跳过: {ports_area_input_txt_path}")
                continue
            print(f"\n\n==================== 开始处理: {ports_area_input_txt_path} ====================")

            start_time = time.time()

            # dis = 40
            # timer = 12

            # 从 folder_name (例如 "5-1") 中提取 id
            current_id = folder_name  # 或者更精确地提取数字部分，例如 i
            # export_result_path = '/home/eda220806/project/verify_results' # 已被 export_result_base_path 替代
            # id = ports_area_input_txt_path.split('.')[0].split('_')[-1] # 这行不再适用，我们用 current_id
            print("==================== Parameters ====================")
            print("torch-version:", torch.__version__)
            print("id:", current_id)  # 使用 current_id
            print("ports_area_input_txt_path:", ports_area_input_txt_path)
            print("ports_link_input_txt_path:", ports_link_input_txt_path)
            print("output_path:", output_path)

            try:  # 增加异常处理，以便一个文件出错不影响后续文件
                item_list, rule_dict, area_list = CzgFunction2.read_data_port_area(ports_area_input_txt_path)
                link_list = CzgFunction2.read_data_port_link(ports_link_input_txt_path)
                instance = Instance(item_list, link_list, rule_dict, area_list)

                timer = len(item_list) / 3.0

                if len(item_list) <= 5:
                    dis = 26
                elif len(item_list) <= 10:
                    dis = 30
                elif len(item_list) <= 25:
                    dis = 38
                elif len(item_list) <= 50:
                    dis = 40
                else:
                    dis = 45

                # 标准化边界
                min_x, min_y = None, None
                for x, y in instance.area:
                    if min_x is None:
                        min_x = x
                    else:
                        min_x = min(min_x, x)
                    if min_y is None:
                        min_y = y
                    else:
                        min_y = min(min_y, y)
                for i in range(len(instance.area)):
                    instance.area[i] = [instance.area[i][0] - min_x, instance.area[i][1] - min_y]

                # 对模块进行放大
                for item in instance.items:
                    for rotate_item in item.rotate_items:
                        rotate_item.boundary = WskhFunction.zoom_use_exhaustive_method(rotate_item, dis)
                        rotate_item.init_rotate_item()

                # 1.按照面积递减排序
                instance.items.sort(key=functools.cmp_to_key(items_compare_by_area_decrease))
                # 2.按照最长边递减
                # instance.items.sort(key=functools.cmp_to_key(items_compare_by_area_longest_line))
                # 3.按照多边形难度递减
                # instance.items.sort(key=functools.cmp_to_key(items_compare_by_area_polygon_difficulty))

                # 传入实例，生成算法对象
                algo = VNSAlgorithm_C_R(instance)

                # 初始序列（测试阶段，直接按照升序作为初始顺序，后面用机器学习方法生成较好的初始序列）
                init_sequence = [i for i in range(len(instance.items))]

                # 调用算法对象对实例进行求解
                print("Running...")
                best_evaluation, my_result, vns_evaluate_cnt, evaluation_total_time = algo.solve(init_sequence.copy(),
                                                                                                 timer,
                                                                                                 min_x, min_y, start_time, GCN_ON, Classify_ON, generate_layoutForGCNTrain_txt_output_path)

                # 程序结束，打印最佳结果
                end_time = time.time()
                print("==================== Solved ====================")
                print("Number of modules placed:", len(best_evaluation.result_list))

                # GCN最后求解一次布线得分才是真正的得分
                items = []
                result_list = best_evaluation.result_list
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
                    copy_rsult = copy.deepcopy(result)

                    copy_rsult.rotate_item.boundary = WskhFunction.zoom_use_exhaustive_method(copy_rsult.rotate_item,
                                                                                              -dis)
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
                loc_area = [(x + min_x, y + min_y) for x, y in instance.area]

                # items是全部的item集合，但是不一定全部都放得下所以需要对links进行过滤，传进去算法的links是过滤后的
                # 传old_links是为了计算all_items_count 为了评分的时候能知道总共有多少个模块
                links = []
                for link in instance.links:
                    links.append(link.link_dict)
                real_links = DataUtils().filter_invalid_links(links, items)
                time_route, real_score = RoutingScore_Interface.Solve_Interface(items, loc_area, real_links, links)

                print("===real_score:", real_score, "===")

                # 需要统计保存的数据如下输出
                score = best_evaluation.obj_value
                vns_iterations = my_result.epochs_size
                evaluation_count = vns_evaluate_cnt
                if vns_evaluate_cnt > 0:
                    avg_eval_time = evaluation_total_time / vns_evaluate_cnt
                else:
                    avg_eval_time = 0
                # avg_eval_time = evaluation_total_time / vns_evaluate_cnt
                total_compute_time = round(end_time - start_time, 2)
                print("得分:", best_evaluation.obj_value)
                print("VNS迭代次数:", my_result.epochs_size)
                print("评估次数:", vns_evaluate_cnt)
                # print("评估总耗时:", evaluation_total_time)
                if vns_evaluate_cnt > 0:
                    print("一次评估平均耗时:", avg_eval_time)
                else:
                    print("一次评估平均耗时: 0")
                # print("一次评估平均耗时:", evaluation_total_time / vns_evaluate_cnt)
                # Todo 这里把最后一次布线时间也算上了
                print("总计算时间:", total_compute_time, "s")

                RedaFileUtils.save_GCNresults_to_csv(output_path, ports_area_input_txt_path, score, real_score, vns_iterations,
                                                  evaluation_count, avg_eval_time, total_compute_time)




                print(f"==================== 完成处理: {ports_area_input_txt_path} ====================")

                # ==================== 核心修改：优先捕获超时异常 ====================
            except TimeoutException as e:
                end_time = time.time()
                total_compute_time = round(end_time - start_time, 2)
                print(f"处理文件 {ports_area_input_txt_path} 时发生超时: {e}")
                print(f"已运行 {total_compute_time} 秒。")

                items = []
                result_list = best_evaluation.result_list
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
                    copy_rsult = copy.deepcopy(result)

                    copy_rsult.rotate_item.boundary = WskhFunction.zoom_use_exhaustive_method(copy_rsult.rotate_item,
                                                                                              -dis)
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
                loc_area = [(x + min_x, y + min_y) for x, y in instance.area]

                # items是全部的item集合，但是不一定全部都放得下所以需要对links进行过滤，传进去算法的links是过滤后的
                # 传old_links是为了计算all_items_count 为了评分的时候能知道总共有多少个模块
                links = []
                for link in instance.links:
                    links.append(link.link_dict)
                real_links = DataUtils().filter_invalid_links(links, items)
                time_route, real_score = RoutingScore_Interface.Solve_Interface(items, loc_area, real_links, links)

                print("===real_score:", real_score, "===")

                # 初始化默认超时结果
                score = 0
                vns_iterations = 0
                evaluation_count = 0
                avg_eval_time = 0

                # 检查algo对象，看是否在超时前已经找到了部分解
                if hasattr(algo, 'best_evaluation') and algo.best_evaluation is not None:
                    print("超时，但已找到部分布局解，记录当前最佳结果...")

                    # 从对象中安全地提取我们能拿到的数据
                    score = algo.best_evaluation.obj_value
                    evaluation_count = algo.vns_evaluate_cnt
                    vns_iterations = algo.vns_iteration_count_realtime  # 读取我们新增的实时迭代次数

                    if evaluation_count > 0 and hasattr(algo, 'evaluation_total_time'):
                        avg_eval_time = algo.evaluation_total_time / evaluation_count

                    print(f"布局得分为: {score}, 评估次数为: {evaluation_count}, VNS迭代次数: {vns_iterations}")
                else:
                    print("超时，且未找到任何可行解，记录为0...")

                # 使用与成功时相同的函数，写入获取到的（部分的）结果
                RedaFileUtils.save_GCNresults_to_csv(output_path, ports_area_input_txt_path,
                                                     score=score,
                                                     real_score=real_score,
                                                     vns_iterations=vns_iterations,
                                                     evaluation_count=evaluation_count,
                                                     avg_eval_time=avg_eval_time,
                                                     total_compute_time=total_compute_time)
                continue

            # ==================== 保留通用的异常捕获，作为最后的保障 ====================

            except Exception as e:
                print(f"处理文件 {ports_area_input_txt_path} 时发生错误: {e}")
                # 你可以选择在这里记录错误到日志文件，或者简单地继续处理下一个文件
                continue  # 继续下一个循环迭代

