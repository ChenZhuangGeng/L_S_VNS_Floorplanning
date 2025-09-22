import re
from itertools import count
import csv
import os

import math
from pathlib import Path

def create_file_from_skip_message(ports_area_input_txt_path: str, output_directory: str) -> str | None:
    """
    根据 "文件不存在，跳过: <路径>" 的消息创建一个 .txt 文件。

    文件名将是这条消息本身（经过处理以移除非法字符），并添加 .txt 后缀。
    文件内容将是 ports_area_input_txt_path。

    :param ports_area_input_txt_path: 原始的、未找到的文件路径字符串。
    :param output_directory: 新创建的 .txt 文件将要保存到的目录。
    :return: 如果成功，返回创建的文件的完整路径；否则返回 None。
    """
    try:
        # 1. 构建文件名基础字符串
        filename_base_string = f"文件不存在，跳过: {ports_area_input_txt_path}"

        # 2. 清理文件名字符串，移除或替换非法字符
        # Windows 文件名非法字符: < > / \ | : " * ?
        # 同时，路径分隔符 (\, /) 和冒号 (:) 肯定会在 ports_area_input_txt_path 中出现
        # 将它们替换为下划线 "_"
        # 使用 re.sub 清理更广泛的非法字符，或者手动替换已知字符

        # 简单替换常见非法字符
        sanitized_filename_base = filename_base_string.replace(":", "_")
        sanitized_filename_base = sanitized_filename_base.replace("/", "_")
        sanitized_filename_base = sanitized_filename_base.replace("\\", "_")
        sanitized_filename_base = sanitized_filename_base.replace("*", "_")
        sanitized_filename_base = sanitized_filename_base.replace("?", "_")
        sanitized_filename_base = sanitized_filename_base.replace("\"", "_")
        sanitized_filename_base = sanitized_filename_base.replace("<", "_")
        sanitized_filename_base = sanitized_filename_base.replace(">", "_")
        sanitized_filename_base = sanitized_filename_base.replace("|", "_")

        # 还可以考虑去除字符串两端的空格或特定字符
        sanitized_filename_base = sanitized_filename_base.strip()

        # 限制文件名长度 (可选，但推荐)
        # 大多数操作系统对文件名长度有限制 (例如 255 个字符或字节)
        # 如果原始路径很长，这里可能需要截断
        max_len = 200  # 为 .txt 和其他可能的路径部分留出余地
        if len(sanitized_filename_base) > max_len:
            sanitized_filename_base = sanitized_filename_base[:max_len] + "_truncated"

        # 3. 添加 .txt 后缀
        final_filename = f"{sanitized_filename_base}.txt"

        # 4. 确保输出目录存在
        os.makedirs(output_directory, exist_ok=True)

        # 5. 构建完整的文件路径
        full_file_path = os.path.join(output_directory, final_filename)

        # 6. 创建并写入文件 (文件内容为原始路径)
        with open(full_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Original path: {ports_area_input_txt_path}\n")
            f.write("This file was created because the above path was not found.\n")

        print(f"提示文件已创建: {full_file_path}")
        return full_file_path

    except Exception as e:
        print(f"创建文件时发生错误: {e}")
        return None



import os
import csv

def save_GCNresults_to_csv(
        output_csv_path,
        case_name_source_path,
        score,
        real_score,  # 新增的参数
        vns_iterations,
        evaluation_count,
        avg_eval_time,
        total_compute_time
):
    """
    将包含GCN预测得分和真实得分的结果保存到CSV文件中。

    Args:
        output_csv_path (str): 输出CSV文件的路径。
        case_name_source_path (str): 用于提取案例名称的源文件路径。
        score (float): 模型的预测得分 (GCN Score)。
        real_score (float): 经过评估的真实得分 (Real Score)。
        vns_iterations (int): VNS算法的迭代次数。
        evaluation_count (int): 评估函数的调用次数。
        avg_eval_time (float): 单次评估的平均耗时。
        total_compute_time (float): 算法总计算时间。
    """
    main_case_name = "N/A"
    sub_case_name_original = "N/A"

    try:
        norm_path = os.path.normpath(case_name_source_path)
        parts = norm_path.split(os.sep)

        if len(parts) >= 3:
            case_name = os.path.join(parts[-3], parts[-2])
        elif len(parts) == 2:
            case_name = parts[-2]
        elif len(parts) == 1 and parts[0] != '':
            case_name = parts[0]
        else:
            case_name = "unknown_case"

        if case_name not in ["unknown_case", "error_extracting_case_name"] and (
                os.sep in case_name or '/' in case_name):
            temp_case_name = case_name.replace('\\', '/')
            if '/' in temp_case_name:
                split_parts = temp_case_name.rsplit('/', 1)
                if len(split_parts) == 2:
                    main_case_name = split_parts[0]
                    sub_case_name_original = split_parts[1]
                else:
                    main_case_name = temp_case_name
                    sub_case_name_original = "N/A"
            else:
                main_case_name = temp_case_name
                sub_case_name_original = "N/A"
        elif case_name not in ["unknown_case", "error_extracting_case_name"]:
            main_case_name = case_name
            sub_case_name_original = "N/A"

    except Exception:
        case_name = "error_extracting_case_name"

    write_header = not os.path.isfile(output_csv_path) or os.path.getsize(output_csv_path) == 0

    with open(output_csv_path, mode='a', newline='', encoding='utf-8') as csvfile:
        # 1. 修改：在表头中增加 '真实得分'
        fieldnames = ['案例名', '主案例名', '子案例名', '得分', '真实得分', 'VNS迭代次数', '评估次数', '一次评估平均耗时',
                      '总计算时间']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        # 处理子案例名以兼容Excel的逻辑保持不变
        sub_case_to_write = sub_case_name_original
        if sub_case_name_original != "N/A" and "-" in sub_case_name_original:
            sub_case_to_write = f"'{sub_case_name_original}"

        # 2. 修改：在写入行中增加 real_score 的数据
        writer.writerow({
            '案例名': case_name,
            '主案例名': main_case_name,
            '子案例名': sub_case_to_write,
            '得分': f"{score:.4f}",
            '真实得分': f"{real_score:.4f}",  # 新增的数据列
            'VNS迭代次数': vns_iterations,
            '评估次数': evaluation_count,
            '一次评估平均耗时': f"{avg_eval_time:.4f}",
            '总计算时间': f"{total_compute_time:.4f}"
        })
    print(f"结果已保存到: {output_csv_path}")


def save_results_to_csv(
        output_csv_path,
        case_name_source_path,
        score,
        vns_iterations,
        evaluation_count,
        avg_eval_time,
        total_compute_time
):
    main_case_name = "N/A"
    sub_case_name_original = "N/A"  # 存储原始的子案例名

    try:
        norm_path = os.path.normpath(case_name_source_path)
        parts = norm_path.split(os.sep)

        if len(parts) >= 3:
            case_name = os.path.join(parts[-3], parts[-2])
        elif len(parts) == 2:
            case_name = parts[-2]
        elif len(parts) == 1 and parts[0] != '':
            case_name = parts[0]
        else:
            case_name = "unknown_case"

        if case_name not in ["unknown_case", "error_extracting_case_name"] and (
                os.sep in case_name or '/' in case_name):
            temp_case_name = case_name.replace('\\', '/')
            if '/' in temp_case_name:
                split_parts = temp_case_name.rsplit('/', 1)
                if len(split_parts) == 2:
                    main_case_name = split_parts[0]
                    sub_case_name_original = split_parts[1]  # 保存原始值
                else:
                    main_case_name = temp_case_name
                    sub_case_name_original = "N/A"  # 如果拆分不符合预期
            else:
                main_case_name = temp_case_name
                sub_case_name_original = "N/A"  # 如果没有分隔符
        elif case_name not in ["unknown_case", "error_extracting_case_name"]:
            main_case_name = case_name
            sub_case_name_original = "N/A"  # 如果 case_name 是单个名称

    except Exception:
        case_name = "error_extracting_case_name"
        # main_case_name 和 sub_case_name_original 保持 "N/A"

    write_header = False
    if not os.path.isfile(output_csv_path):
        write_header = True
    elif os.path.getsize(output_csv_path) == 0:
        write_header = True

    with open(output_csv_path, mode='a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['案例名', '主案例名', '子案例名', '得分', 'VNS迭代次数', '评估次数', '一次评估平均耗时',
                      '总计算时间']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        # 为了让 Excel 将其识别为文本，可以在前面加一个单引号 '
        # 或者使用 ="内容" 的格式，例如 sub_case_excel_friendly = f'="{sub_case_name_original}"'
        # 这里采用加单引号的方式，比较常见且简单
        sub_case_excel_friendly = f"'{sub_case_name_original}" if sub_case_name_original != "N/A" else "N/A"
        # 如果你不想在CSV文件中看到这个单引号，而只想影响Excel的解析，
        # 那么 ="内容" 的方式可能更好，但单引号在CSV中是不可见的（Excel处理后）
        # 或者，如果sub_case_name_original本身就是纯数字或可能被误解的，才加标记。
        # 例如，如果 "5-1" 这种形式，可以加标记。
        # 考虑到 "5-1" 这种形式，我们这里统一为子案例名加上标记
        # 如果原始值是 "N/A"，则不加单引号

        # 一个更安全的做法是只对可能被Excel误解的格式进行处理
        # 但为了简单，这里如果不是N/A就处理
        if sub_case_name_original != "N/A" and "-" in sub_case_name_original:  # 假设带"-"的才需要特殊处理
            sub_case_to_write = f"'{sub_case_name_original}"  # Excel中会显示为 5-1
        # 或者 sub_case_to_write = f'="{sub_case_name_original}"' # Excel中会显示为 5-1
        else:
            sub_case_to_write = sub_case_name_original

        writer.writerow({
            '案例名': case_name,
            '主案例名': main_case_name,
            '子案例名': sub_case_to_write,  # 使用处理后的子案例名
            '得分': f"{score:.4f}",
            'VNS迭代次数': vns_iterations,
            '评估次数': evaluation_count,
            '一次评估平均耗时': f"{avg_eval_time:.4f}",
            '总计算时间': f"{total_compute_time:.4f}"
        })
    print(f"结果已保存到: {output_csv_path}")


def parse_links_file(file_path):
    links = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Link"):
                link_dict = {}
                modules_line = lines[i + 1].strip().split()
                values_line = lines[i + 2].strip().split()
                for module, value in zip(modules_line, values_line):
                    link_dict[module] = value
                links.append(link_dict)
                i += 3
            else:
                i += 1
    return links


def parse_items_file(file_path):
    from src.classes.EdaClasses import Item

    items = []
    current_name = ""
    current_boundary = []
    current_ports = []
    current_boundary_layer = ""
    current_port_layers = []

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            area_match = re.match(r'Area:\(([^)]+)\)\(([^)]+)\)\(([^)]+)\)\(([^)]+)\)', lines[0])
            area = tuple(tuple(map(float, coord.split(','))) for coord in area_match.groups())

            for line in lines[1:]:
                line = line.strip()
                if line.startswith("Module:"):
                    if current_name:

                        item = Item(
                            name=current_name,
                            boundary=current_boundary,
                            ports=current_ports,
                            boundary_layer=current_boundary_layer,
                            port_layers=current_port_layers
                        )
                        items.append(item)

                    current_name = line.split(":")[1].strip()
                    current_boundary = []
                    current_ports = []
                    current_boundary_layer = ""
                    current_port_layers = []
                elif line.startswith("Boundary:"):
                    boundary_coords = re.findall(r'\(([^)]+)\)', line)
                    current_boundary_layer = line.split(";")[-1].strip()
                    current_boundary = [list(map(float, coord.split(','))) for coord in boundary_coords]
                elif line.startswith("Port:"):
                    port_coords = re.findall(r'\(([^)]+)\)', line)
                    port_type = line.split(";")[-1].strip()
                    port_info = {'coordinates': [tuple(map(float, coord.split(','))) for coord in port_coords],
                                 'type': port_type}
                    current_ports.append(port_info)
                    current_port_layers.append(port_type)

            if current_name:
                item = Item(
                    name=current_name,
                    boundary=current_boundary,
                    ports=current_ports,
                    boundary_layer=current_boundary_layer,
                    port_layers=current_port_layers
                )
                items.append(item)
    except FileNotFoundError:
        print(f"错误：布局文件 {file_path} 未找到。将跳过此项。")
        return None, None  # 返回 None, None 表示文件未找到

    except Exception as e:  # 捕获其他可能的解析错误
        print(f"解析文件 {file_path} 时发生错误: {e}")
        return None, None  # 表示其他解析失败
    return area, items

def read_WSKH_Result_file(file_path):
    """
    读取指定格式的txt文件，并返回模块信息列表。

    参数:
    file_path (str): 要读取的txt文件路径

    返回:
    list: 包含模块信息的字典列表，每个字典包含Module、Orient和Position键
    """
    modules = []
    with open(file_path, 'r') as file:
        # 读取并过滤空行
        lines = [line.strip() for line in file if line.strip()]

        # 检查行数是否为3的倍数
        if len(lines) % 3 != 0:
            raise ValueError("文件格式错误：行数不是3的倍数，可能存在数据不完整")

        # 分块处理每三个行
        for i in range(0, len(lines), 3):
            chunk = lines[i:i + 3]
            module_info = {}
            for line in chunk:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                if key == 'Module':
                    module_info['Module'] = value
                elif key == 'Orient':
                    module_info['Orient'] = value
                elif key == 'Position':
                    # 提取坐标并转换为浮点数元组
                    pos_str = value.strip('()')
                    x, y = map(float, pos_str.split(','))
                    module_info['Position'] = (x, y)
                else:
                    raise ValueError(f"未知键 '{key}' 在文件中")

            # 验证必需字段是否存在
            required_keys = {'Module', 'Orient', 'Position'}
            if not required_keys.issubset(module_info.keys()):
                missing = required_keys - module_info.keys()
                raise ValueError(f"模块信息不完整，缺失字段: {missing}")

            modules.append(module_info)

    return modules


def calculate_center(boundary):
    """
    计算正交多边形包络矩形的中心点

    参数:
    boundary (list): 由坐标点组成的列表，格式为[[x1,y1], [x2,y2], ...]

    返回:
    tuple: (center_x, center_y) 中心点坐标，保留两位小数
    """
    # 提取所有x和y坐标
    x_coords = [point[0] for point in boundary]
    y_coords = [point[1] for point in boundary]

    # 计算包络矩形边界
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    # 计算中心点坐标
    center_x = round((x_min + x_max) / 2, 2)
    center_y = round((y_min + y_max) / 2, 2)

    return (center_x, center_y)


def rotate_point(x, y, cx, cy, angle_deg):
    """
    绕中心点(cx, cy)顺时针旋转指定角度
    参数:
        x, y: 待旋转点坐标
        cx, cy: 旋转中心
        angle_deg: 顺时针旋转角度（度）
    返回:
        (new_x, new_y): 旋转后坐标，保留两位小数
    """
    angle_rad = math.radians(-angle_deg)  # 转换为逆时针弧度
    dx = x - cx
    dy = y - cy
    new_x = dx * math.cos(angle_rad) - dy * math.sin(angle_rad) + cx
    new_y = dx * math.sin(angle_rad) + dy * math.cos(angle_rad) + cy
    return (round(new_x, 2), round(new_y, 2))


def transform_items_coordinates(items, modules):
    """
    根据modules中的坐标信息转换items的坐标（包含旋转）
    参数:
        items: 原始item列表，每个item需包含name/boundary/ports属性
        modules: 从文件读取的module列表，每个module包含Module/Orient/Position
    返回:
        转换后的items列表（原地修改）
    """
    module_map = {m['Module']: m for m in modules}

    for item in items:
        mod = module_map.get(item.name)
        if not mod:
            continue

        original_center = calculate_center(item.boundary)
        target_center = mod['Position']
        dx = target_center[0] - original_center[0]
        dy = target_center[1] - original_center[1]

        # 平移操作
        item.boundary = [[x + dx, y + dy] for x, y in item.boundary]
        for port in item.ports:
            port['coordinates'] = [(x + dx, y + dy) for x, y in port['coordinates']]

        # 旋转操作
        rotation_angle = int(mod['Orient'][1:])  # 提取R后面的数字
        if rotation_angle != 0:
            cx, cy = target_center
            item.boundary = [rotate_point(x, y, cx, cy, rotation_angle) for x, y in item.boundary]
            for port in item.ports:
                port['coordinates'] = [rotate_point(x, y, cx, cy, rotation_angle) for x, y in port['coordinates']]

    return items


def add_evaluation_count_to_path(original_path: str, evaluation_count: int) -> str:
    """
    修改文件路径，在 'sample-i' 格式的文件夹名后追加评估次数。

    例如：
    - 输入路径: '.../layoutForGCNTrain/5-1/placement_info.txt'
    - 评估次数: 3
    - 输出路径: '.../layoutForGCNTrain/5-1-3/placement_info.txt'

    参数:
    original_path (str): 原始的文件路径字符串。
    evaluation_count (int): 要追加的评估次数。

    返回:
    str: 修改后的新文件路径字符串。
    """
    if original_path==None:
        return ""
    if not original_path:
        return ""

    # 1. 将字符串路径转换为 Path 对象，以便轻松操作
    p = Path(original_path)

    # 2. 获取路径的各个部分。
    #    p.parts 对于 'D:/.../5-1/info.txt' 会是 ('D:\\', '...', '5-1', 'info.txt')
    #    我们要修改的是倒数第二个部分。
    parts = list(p.parts)

    # 3. 定位并修改目标文件夹名
    #    我们假设目标文件夹总是倒数第二个部分
    if len(parts) >= 2:
        target_folder_name = parts[-2]
        # 在原文件夹名后追加 "-次数"
        new_folder_name = f"{target_folder_name}-{evaluation_count}"
        parts[-2] = new_folder_name
    else:
        # 如果路径太短，无法应用逻辑，则直接返回原路径
        return original_path

    # 4. 使用修改后的部分重新组合成一个新路径
    #    Path(*parts) 会根据操作系统正确地拼接路径
    new_path = Path(*parts)

    # 5. 将 Path 对象转换回字符串并返回
    return str(new_path)


def read_score_from_file(score_file: str) -> float | None:
    """
    安全地从一个文本文件中读取第一行的浮点数分数。

    这个函数包含了完整的错误处理，能够应对多种异常情况。

    参数:
    score_file (str): 指向 score.txt 文件的完整路径。

    返回:
    float: 如果成功读取并转换，返回文件中的浮点数分数。
    None: 如果文件不存在、文件为空、第一行不是有效的数字或发生任何其他错误。
    """
    # 检查路径是否为有效字符串
    if not isinstance(score_file, str) or not score_file:
        print("错误: 提供的文件路径无效。")
        return None

    try:
        # 使用 'with open' 语句确保文件在使用后会被自动关闭，这是最佳实践
        with open(score_file, 'r', encoding='utf-8') as f:
            # 1. 只读取文件的第一行
            first_line = f.readline()

            # 2. 检查第一行是否存在内容
            if first_line:
                # 3. 使用 .strip() 清除字符串两端的空白字符（如空格、换行符'\n'等）
                #    这是至关重要的一步！
                cleaned_line = first_line.strip()

                # 4. 将清理后的字符串转换为浮点数
                score = float(cleaned_line)
                return score
            else:
                # 如果文件是空的，readline()会返回空字符串
                print(f"警告: 文件 '{score_file}' 为空，无法读取分数。")
                return None

    except FileNotFoundError:
        # 如果文件路径不存在，捕获这个错误
        print(f"错误: 找不到文件 '{score_file}'。")
        return None
    except ValueError:
        # 如果第一行的内容无法被转换成浮点数（例如，里面包含字母），捕获这个错误
        print(f"错误: 文件 '{score_file}' 的第一行内容不是一个有效的数字。")
        return None
    except Exception as e:
        # 捕获其他所有意料之外的错误，增加代码的健壮性
        print(f"读取文件 '{score_file}' 时发生未知错误: {e}")
        return None
def generate_layoutForGCNTrain_txt(loc_area, items, score, layout_filename):
    """
    最终修正版：生成布局文件和分数文件。
    保存每一次迭代的布局结果 和 对应得分 供GCN训练使用
    功能:
    1. 在指定路径下创建布局文件，兼容新的 ports 和 port_layers 数据结构。
    2. 在同一目录下，自动创建 score.txt 文件。
    3. 如果指定的输出目录不存在，会自动创建。
    """

    # 辅助函数，用于将坐标列表格式化为 (x, y)(x, y)... 的字符串
    def _format_coords(coords_list):
        # 如果 coords_list 格式不正确（比如是None），返回空字符串
        if not isinstance(coords_list, (list, tuple)):
            return ""
        return "".join([f"({float(c[0])}, {float(c[1])})" for c in coords_list])

    try:
        # 步骤 1: 路径和目录管理
        directory = os.path.dirname(layout_filename)
        score_filename = os.path.join(directory, "score.txt")

        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            # print(f"已自动创建新文件夹: {directory}")

        # 步骤 2: 生成布局文件
        with open(layout_filename, 'w', encoding='utf-8') as f:
            area_str = f"Area:{_format_coords(loc_area)}"
            f.write(area_str + '\n')

            rule_str = "Rule:GATE(5,5);SD(5,5);GATE_SD(0.5);GATE_ITO(0.5);SD_ITO(0.5)"
            f.write(rule_str + '\n')

            for item in items:
                module_name = getattr(item, 'name', 'UnknownModule')
                f.write(f"Module:{module_name}\n")

                boundary_coords = getattr(item, 'boundary', [])
                boundary_layer = getattr(item, 'boundary_layer', 'N/A')
                boundary_str = f"Boundary:{_format_coords(boundary_coords)};{boundary_layer}"
                f.write(boundary_str + '\n')

                # --- 最终修正点：处理字典形式的 port 列表 ---
                # 获取 ports 列表，其成员现在是字典
                ports_list_of_dicts = getattr(item, 'ports', [])

                # 直接遍历这个字典列表
                for port_dict in ports_list_of_dicts:
                    # 从每个字典中安全地获取坐标和类型
                    # 使用 .get() 方法，如果键不存在则返回默认值（空列表或'N/A'）
                    port_coords = port_dict.get('coordinates', [])
                    port_type = port_dict.get('type', 'N/A')

                    # 格式化并写入文件
                    port_str = f"Port:{_format_coords(port_coords)};{port_type}"
                    f.write(port_str + '\n')
                # --- 修正结束 ---

        print(f"布局文件 '{layout_filename}' 已成功生成。")

        # 步骤 3: 生成分数文件
        with open(score_filename, 'w', encoding='utf-8') as f:
            f.write(str(score))

        print(f"分数文件 '{score_filename}' 已成功生成。")

    except Exception as e:
        print(f"生成文件时发生了一个意料之外的错误: {e}")


def read_and_split_simple_txt(file_path):
    """
    读取一个简单的文本文件，文件每行包含一个'sampleX\Y-Z'格式的字符串。
    将每一行按'\'拆分。

    参数:
        file_path (str): 文本文件的路径。

    返回:
        list: 列表的列表，例如 [['sample5', '5-2'], ['sample5', '5-15']]
    """
    results = []
    try:
        # 对于简单的文本文件，通常用'utf-8'即可
        with open(file_path, mode='r', encoding='utf-8') as infile:
            # 遍历文件的每一行
            for line in infile:
                # 1. strip()用于移除每行末尾的换行符和所有空白
                clean_line = line.strip()

                # 2. 如果是空行，则跳过
                if not clean_line:
                    continue

                # 3. 按'\'拆分并添加到结果中
                parts = clean_line.split('\\')
                results.append(parts)

    except FileNotFoundError:
        print(f"错误：文件未找到，请检查路径 '{file_path}'")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")

    return results




if __name__ == '__main__':
    # Example usage
    links_file = r'D:\本科\比赛\2022.09.01 EDA图像拼接\2024更新布线算法\data\EDA_DATA\connect\connect_file\connect_5.txt'
    item_file = r'D:\本科\比赛\2022.09.01 EDA图像拼接\2024更新布线算法\data\EDA_DATA\sample5\5-1\placement_info.txt'

    # 打印解析后的内容
    links = parse_links_file(links_file)
    area, items = parse_items_file(item_file)

    # 检查所有案例的area是不是全为矩形（是）
    i = 0
    count = 8
    c = [5,10,16,20,25,30,35,40,45]
    try:
        for i in range(600):
            item_file = r'D:\本科\比赛\2022.09.01 EDA图像拼接\2024更新布线算法\data\EDA_DATA\sample'+ str(c[count]) +'\\'+ str(c[count]) + '-' + str(i+1) +'\placement_info.txt'
            area, items = parse_items_file(item_file)
            if len(area) != 4:
                print("item_file:" + item_file)
    except  Exception as e:
        print("Error:", e)
        print("item_file:" + item_file)


    # print("Area:", area)
    # print("Links:", links)
    # print("Items:")
    # for item in items:
    #     print(f"Module: {item.name}")
    #     print(f"Boundary Coordinates: {item.boundary}, Boundary Layer: {item.boundary_layer}")
    #     for port in item.ports:
    #         print(f"Port Coordinates: {port['coordinates']}, Port Layer: {port['type']}")
    #     print(f"Boundary Layer: {item.boundary_layer}")
