import re
import os
import csv
import math
from pathlib import Path

def create_file_from_skip_message(ports_area_input_txt_path: str, output_directory: str) -> str | None:

    try:
        filename_base_string = f"文件不存在，跳过: {ports_area_input_txt_path}"

        sanitized_filename_base = filename_base_string.replace(":", "_")
        sanitized_filename_base = sanitized_filename_base.replace("/", "_")
        sanitized_filename_base = sanitized_filename_base.replace("\\", "_")
        sanitized_filename_base = sanitized_filename_base.replace("*", "_")
        sanitized_filename_base = sanitized_filename_base.replace("?", "_")
        sanitized_filename_base = sanitized_filename_base.replace("\"", "_")
        sanitized_filename_base = sanitized_filename_base.replace("<", "_")
        sanitized_filename_base = sanitized_filename_base.replace(">", "_")
        sanitized_filename_base = sanitized_filename_base.replace("|", "_")

        sanitized_filename_base = sanitized_filename_base.strip()

        max_len = 200
        if len(sanitized_filename_base) > max_len:
            sanitized_filename_base = sanitized_filename_base[:max_len] + "_truncated"

        final_filename = f"{sanitized_filename_base}.txt"

        os.makedirs(output_directory, exist_ok=True)

        full_file_path = os.path.join(output_directory, final_filename)

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

    try:
        output_dir = os.path.dirname(output_csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    except Exception as e:
        print(f"警告：自动创建目录 {output_dir} 失败: {e}")
        pass
    write_header = not os.path.isfile(output_csv_path) or os.path.getsize(output_csv_path) == 0

    with open(output_csv_path, mode='a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['案例名', '主案例名', '子案例名', '得分', '真实得分', 'VNS迭代次数', '评估次数',
                      '一次评估平均耗时',
                      '总计算时间']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        sub_case_to_write = sub_case_name_original
        if sub_case_name_original != "N/A" and "-" in sub_case_name_original:
            sub_case_to_write = f"'{sub_case_name_original}"

        writer.writerow({
            '案例名': case_name,
            '主案例名': main_case_name,
            '子案例名': sub_case_to_write,
            '得分': f"{score:.4f}",
            '真实得分': f"{real_score:.4f}",
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

        if sub_case_name_original != "N/A" and "-" in sub_case_name_original:  # 假设带"-"的才需要特殊处理
            sub_case_to_write = f"'{sub_case_name_original}"  # Excel中会显示为 5-1
        else:
            sub_case_to_write = sub_case_name_original

        writer.writerow({
            '案例名': case_name,
            '主案例名': main_case_name,
            '子案例名': sub_case_to_write,
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
        return None, None

    except Exception as e:
        print(f"解析文件 {file_path} 时发生错误: {e}")
        return None, None
    return area, items

def read_WSKH_Result_file(file_path):
    modules = []
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]

        if len(lines) % 3 != 0:
            raise ValueError("文件格式错误：行数不是3的倍数，可能存在数据不完整")

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
                    pos_str = value.strip('()')
                    x, y = map(float, pos_str.split(','))
                    module_info['Position'] = (x, y)
                else:
                    raise ValueError(f"未知键 '{key}' 在文件中")

            required_keys = {'Module', 'Orient', 'Position'}
            if not required_keys.issubset(module_info.keys()):
                missing = required_keys - module_info.keys()
                raise ValueError(f"模块信息不完整，缺失字段: {missing}")

            modules.append(module_info)

    return modules


def calculate_center(boundary):
    x_coords = [point[0] for point in boundary]
    y_coords = [point[1] for point in boundary]

    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    center_x = round((x_min + x_max) / 2, 2)
    center_y = round((y_min + y_max) / 2, 2)

    return (center_x, center_y)


def rotate_point(x, y, cx, cy, angle_deg):
    angle_rad = math.radians(-angle_deg)
    dx = x - cx
    dy = y - cy
    new_x = dx * math.cos(angle_rad) - dy * math.sin(angle_rad) + cx
    new_y = dx * math.sin(angle_rad) + dy * math.cos(angle_rad) + cy
    return (round(new_x, 2), round(new_y, 2))


def transform_items_coordinates(items, modules):
    module_map = {m['Module']: m for m in modules}

    for item in items:
        mod = module_map.get(item.name)
        if not mod:
            continue

        original_center = calculate_center(item.boundary)
        target_center = mod['Position']
        dx = target_center[0] - original_center[0]
        dy = target_center[1] - original_center[1]

        item.boundary = [[x + dx, y + dy] for x, y in item.boundary]
        for port in item.ports:
            port['coordinates'] = [(x + dx, y + dy) for x, y in port['coordinates']]

        rotation_angle = int(mod['Orient'][1:])
        if rotation_angle != 0:
            cx, cy = target_center
            item.boundary = [rotate_point(x, y, cx, cy, rotation_angle) for x, y in item.boundary]
            for port in item.ports:
                port['coordinates'] = [rotate_point(x, y, cx, cy, rotation_angle) for x, y in port['coordinates']]

    return items


def add_evaluation_count_to_path(original_path: str, evaluation_count: int) -> str:
    if original_path==None:
        return ""
    if not original_path:
        return ""

    p = Path(original_path)

    parts = list(p.parts)

    if len(parts) >= 2:
        target_folder_name = parts[-2]
        new_folder_name = f"{target_folder_name}-{evaluation_count}"
        parts[-2] = new_folder_name
    else:
        return original_path

    new_path = Path(*parts)

    return str(new_path)


def read_score_from_file(score_file: str) -> float | None:
    if not isinstance(score_file, str) or not score_file:
        print("错误: 提供的文件路径无效。")
        return None

    try:
        with open(score_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()

            if first_line:
                cleaned_line = first_line.strip()
                score = float(cleaned_line)
                return score
            else:
                print(f"警告: 文件 '{score_file}' 为空，无法读取分数。")
                return None

    except FileNotFoundError:
        print(f"错误: 找不到文件 '{score_file}'。")
        return None
    except ValueError:
        print(f"错误: 文件 '{score_file}' 的第一行内容不是一个有效的数字。")
        return None
    except Exception as e:
        print(f"读取文件 '{score_file}' 时发生未知错误: {e}")
        return None
def generate_layoutForGCNTrain_txt(loc_area, items, score, layout_filename):
    def _format_coords(coords_list):
        if not isinstance(coords_list, (list, tuple)):
            return ""
        return "".join([f"({float(c[0])}, {float(c[1])})" for c in coords_list])

    try:
        directory = os.path.dirname(layout_filename)
        score_filename = os.path.join(directory, "score.txt")

        if directory and not os.path.exists(directory):
            os.makedirs(directory)

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

                ports_list_of_dicts = getattr(item, 'ports', [])

                for port_dict in ports_list_of_dicts:
                    port_coords = port_dict.get('coordinates', [])
                    port_type = port_dict.get('type', 'N/A')
                    port_str = f"Port:{_format_coords(port_coords)};{port_type}"
                    f.write(port_str + '\n')

        print(f"布局文件 '{layout_filename}' 已成功生成。")
        with open(score_filename, 'w', encoding='utf-8') as f:
            f.write(str(score))
        print(f"分数文件 '{score_filename}' 已成功生成。")
    except Exception as e:
        print(f"生成文件时发生了一个意料之外的错误: {e}")


def read_and_split_simple_txt(file_path):
    results = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as infile:
            for line in infile:
                clean_line = line.strip()

                if not clean_line:
                    continue

                parts = clean_line.split('\\')
                results.append(parts)

    except FileNotFoundError:
        print(f"错误：文件未找到，请检查路径 '{file_path}'")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")

    return results


