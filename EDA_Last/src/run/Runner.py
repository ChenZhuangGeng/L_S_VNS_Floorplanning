import argparse
import src.function.CzgFunction2 as CzgFunction2
import src.function.ReadFileUtils as RedaFileUtils
from src.model.ClassificationAndRegression.HeuModel import *
import torch
import time
import functools
from src.function.RoutingScore_LAHC import TimeoutException
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning, module='sklearn')

def set_global_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    except ImportError:
        pass
    os.environ['PYTHONHASHSEED'] = str(seed)


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

    parser = argparse.ArgumentParser(description="处理 EDA 数据样本。")
    parser.add_argument("--start", type=int, default=1, help="处理的起始索引。")
    parser.add_argument("--end", type=int, default=3, help="处理的结束索引 (不包含此索引)。")

    args = parser.parse_args()

    start = args.start
    end = args.end

    current_script_path = os.path.abspath(__file__)
    run_dir = os.path.dirname(current_script_path)
    src_root_dir = os.path.dirname(run_dir)
    project_root = os.path.dirname(src_root_dir)

    sample5_45 = [5,10,16,20,25,30,40,45]
    GCN_ON = False
    Classify_ON = False # 默认关闭二分类 弃用
    id = f"{start}_{end}"

    for sample in sample5_45:
        base_path = os.path.join(project_root, 'src', 'data', 'EDA_DATA', 'sample' + str(sample))

        ports_link_input_txt_path = os.path.join(project_root, 'src', 'data', 'EDA_DATA', 'connect', 'connect_file',
                                                 'connect_' + str(sample) + '.txt')

        # --- 输出文件路径 (相对于项目根目录) ---
        output_path = os.path.join(project_root, 'src', 'result', 'RealRouting', 'result_'+ id + '.csv')

        skip_path_logdir = os.path.join(project_root, 'src', 'result', 'RealRouting', 'skip_logs', 'sample' + str(sample))

        for i in range(start, end):
            folder_name = f"{sample}-{i}"
            ports_area_input_txt_path = os.path.join(base_path, folder_name, 'placement_info.txt')

            generate_layoutForGCNTrain_txt_output_path = f"src/temp/{folder_name}/placement_info.txt"

            # 检查文件是否存在，如果不存在则跳过本次循环
            if not os.path.exists(ports_area_input_txt_path):
                RedaFileUtils.create_file_from_skip_message(folder_name, skip_path_logdir)
                print(f"文件不存在，跳过: {ports_area_input_txt_path}")
                continue
            print(f"\n\n==================== 开始处理: {ports_area_input_txt_path} ====================")

            start_time = time.time()


            current_id = folder_name
            print("==================== Parameters ====================")
            print("torch-version:", torch.__version__)
            print("id:", current_id)
            print("ports_area_input_txt_path:", ports_area_input_txt_path)
            print("ports_link_input_txt_path:", ports_link_input_txt_path)
            print("output_path:", output_path)

            try:
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

                for item in instance.items:
                    for rotate_item in item.rotate_items:
                        rotate_item.boundary = WskhFunction.zoom_use_exhaustive_method(rotate_item, dis)
                        rotate_item.init_rotate_item()

                instance.items.sort(key=functools.cmp_to_key(items_compare_by_area_decrease))

                algo = VNSAlgorithm_C_R(instance)

                init_sequence = [i for i in range(len(instance.items))]

                print("Running...")
                best_evaluation, my_result, vns_evaluate_cnt, evaluation_total_time = algo.solve(init_sequence.copy(),
                                                                                                 timer,
                                                                                                 min_x, min_y, start_time, GCN_ON, Classify_ON, generate_layoutForGCNTrain_txt_output_path)

                print("==================== Solved ====================")
                print("Number of modules placed:", len(best_evaluation.result_list))
                end_time = time.time()

                score = best_evaluation.obj_value
                vns_iterations = my_result.epochs_size
                evaluation_count = vns_evaluate_cnt
                if vns_evaluate_cnt > 0:
                    avg_eval_time = evaluation_total_time / vns_evaluate_cnt
                else:
                    avg_eval_time = 0
                total_compute_time = round(end_time - start_time, 2)
                print("得分:", best_evaluation.obj_value)
                print("VNS迭代次数:", my_result.epochs_size)
                print("评估次数:", vns_evaluate_cnt)
                print("评估总耗时:", evaluation_total_time)
                if vns_evaluate_cnt > 0:
                    print("一次评估平均耗时:", avg_eval_time)
                else:
                    print("一次评估平均耗时: 0")
                print("总计算时间:", total_compute_time, "s")

                RedaFileUtils.save_results_to_csv(output_path, ports_area_input_txt_path, score, vns_iterations,
                                                  evaluation_count, avg_eval_time, total_compute_time)


                print(f"==================== 完成处理: {ports_area_input_txt_path} ====================")


            except TimeoutException as e:
                end_time = time.time()
                total_compute_time = round(end_time - start_time, 2)
                print(f"处理文件 {ports_area_input_txt_path} 时发生超时: {e}")
                print(f"已运行 {total_compute_time} 秒。")
                # 默认超时结果为0
                score = 0
                vns_iterations = 0
                evaluation_count = 0
                avg_eval_time = 0

                if hasattr(algo, 'best_evaluation') and algo.best_evaluation is not None:
                    print("超时，但已找到部分解，记录当前最佳结果...")
                    score = algo.best_evaluation.obj_value
                    evaluation_count = algo.vns_evaluate_cnt
                    vns_iterations = algo.vns_iteration_count_realtime
                    if hasattr(algo, 'evaluation_total_time') and evaluation_count > 0:
                        avg_eval_time = algo.evaluation_total_time / evaluation_count
                    print(f"得分为: {score}, 评估次数为: {evaluation_count}")

                else:
                    print("超时，且未找到任何可行解，记录为0...")
                RedaFileUtils.save_results_to_csv(output_path, ports_area_input_txt_path,
                                                  score=score,
                                                  vns_iterations=vns_iterations,
                                                  evaluation_count=evaluation_count,
                                                  avg_eval_time=avg_eval_time,
                                                  total_compute_time=total_compute_time)
                continue


            except Exception as e:
                print(f"处理文件 {ports_area_input_txt_path} 时发生错误: {e}")
                continue