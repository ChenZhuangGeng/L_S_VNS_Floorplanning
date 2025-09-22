import copy
from datetime import datetime
import heapq
import random
import re
import ast
import collections
import math
import os
import time

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon
from shapely.geometry import Point
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import sys # 用于退出程序
from pylab import mpl

from src.classes.EdaClasses import RotateItem, Item
from src.function import ReadFileUtils


class TimeoutException(Exception):
    """自定义一个超时异常，用于从深层循环中跳出。"""
    pass

# --- RoutingScore 类 ---
class RoutingScore():
    def __init__(self, items, area, links):
        """
        构造函数 - 用于初始化 RoutingScore 实例。
        """
        self.items = items
        self.area = area
        self.links = links
        # 可以在这里预计算一些不变量，如果需要的话

    def run_for_getOneSolution(self, area, items, source_list_grid, target_list_grid, grid_list, weight_list, links, params, modulelist_Target, modulelist_Source, start_time, max_time):

        # grid_raw_copy_for_plot = self.set_grid_weights(area, items, params['grid_size'])


        all_paths = []  # 初始化所有路径列表
        miss_items = []
        success_connect_count = 0
        for i in range(len(grid_list)):
            link = links[i]
            grid = grid_list[i]
            weight = weight_list[i]
            miss_items_temp = []
            # 画障碍图
            # DataUtils.plot_grid_with_path(grid, [])
            # temp_grid_for_plot = grid.copy()

            for path in all_paths:
                grid = self.update_grid(grid, path)
                weight = self.update_weight(weight, path, params['grid_size'], params['affected_radius'], params['affected_value_for_path'],
                                          params['used_value'])

            grid, weights, paths = self.multi_source_region_search_V2(grid, weight, source_list_grid[i],
                                                                    target_list_grid[i], start_time, max_time)
            # print("path ", paths)
            # 如果paths的长度和source_list_grid[i]的长度不一样，就说明有遗漏的item没连上
            if len(paths) != len(source_list_grid[i]):
                # 有漏连item，但是v3-2这个版本是连坐整个link的都统计上
                miss_items_temp.extend(modulelist_Target[i])
                miss_items_temp.extend(modulelist_Source[i])
                # 清空当前路径
                paths = []
            else:
                success_connect_count += 1
            miss_items.append(miss_items_temp)
            all_paths.append(paths)

            # 画障碍图与连线情况
            # DataUtils.plot_grid_with_path(temp_grid_for_plot, all_paths)
        # 对miss_items进行整理去重
        flat_list = [item for sublist in miss_items if sublist for item in sublist]
        miss_items = list(dict.fromkeys(flat_list))
        # DataUtils.plot_grid_with_path(grid_raw_copy_for_plot, all_paths)

        # 为了计算评分，需要如下输出
        # 输出为总共的模块list、以及对应的面积list、布局成功的模块list、以及对应的面积list、成功模块和布线围成的面积、成功模块的总布线长度、布线连接成功率、线长路径和没连上的items
        # 计算两个对应的面积list
        connected_items = DataUtils.find_connected_items(items, miss_items)
        items_area_list = DataUtils.calculate_item_areas(items)
        connected_items_area_list = DataUtils.calculate_item_areas(connected_items)

        area_min = area[0]
        area_max = area[2]
        original_path_coords = DataUtils.convert_paths_list_from_grid(all_paths, area_min, area_max, params['grid_size'])
        success_area_size = DataUtils.calculate_overall_bounding_box_area(connected_items, original_path_coords)
        success_path_length = 0.0
        for paths in original_path_coords:
            for path in paths:
                if len(path) < 2:  # 如果路径长度小于2，跳过
                    continue
                for i in range(len(path) - 1):
                    point1 = path[i]
                    point2 = path[i + 1]
                    success_path_length += abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])  # 曼哈顿距离

        SuccessRate_WiringConnection = success_connect_count / len(links)


        return items, items_area_list, connected_items, connected_items_area_list, success_area_size, success_path_length, SuccessRate_WiringConnection, all_paths, miss_items
        # return all_paths,miss_items

    # --- run_for_random 函数已移除 ---

    def convert_to_grid_coordinates(self, x, y, area_min, area_max, grid_size):
        """
        将实际坐标转换为网格坐标（行，列）。
        """
        # 先计算原始的浮点型索引
        raw_col = (x - area_min[0]) / grid_size
        raw_row = (area_max[1] - y) / grid_size  # y轴反转

        # 使用 floor 获取单元格索引
        # 这意味着一个点总是属于其左边/上边的单元格
        col = math.floor(raw_col)
        row = math.floor(raw_row)
        return row, col



    def set_grid_weights(self, area, items, grid_size):
        # Area: ((-20.0, -20.0), (444.0, -20.0), (444.0, 268.0), (-20.0, 268.0))
        # 计算区域宽度和高度
        area_min = area[0]
        area_max = area[2]
        width = area[2][0] - area[0][0]
        height = area[2][1] - area[0][1]
        print("width", width)
        print("height", height)

        # 计算网格行数和列数
        num_cols = int(width / grid_size)
        num_rows = int(height / grid_size)

        # 创建网格（一个二维数组，初始化为0）
        grid = np.zeros((num_rows, num_cols))

        for item in items:
            # 创建多边形路径
            poly_path = Path(item.boundary)

            # 遍历网格并检查是否在多边形内
            for row in range(num_rows):
                for col in range(num_cols):
                    # grid_x = area_min[0] + col * grid_size + grid_size / 2  # 网格中心X
                    # grid_y = area_max[1] - row * grid_size - grid_size / 2  # 网格中心Y（翻转y轴）
                    grid_x = np.round(area_min[0] + col * grid_size + grid_size / 2, decimals=6)
                    grid_y = np.round(area_max[1] - row * grid_size - grid_size / 2, decimals=6)

                    if poly_path.contains_point((grid_x, grid_y)):
                        grid[row, col] = 1  # 标记占据

        # 将网格输出到 txt 文件,暂不需要
        # grid_file = r"D:\IDEA2020\Project\EDA_Last\src\temp\data\case01InputData\grid\grid_output.txt"
        # with open(grid_file, "w") as f:
        #     for row in grid:
        #         f.write(" ".join([f"{int(cell):2d}" for cell in row]) + "\n")

        # 画一下网格点图
        # DataUtils.plot_grid(grid_file)

        # 输出网格大小
        print(f"Grid dimensions: {num_rows}x{num_cols}")

        return grid

    def set_grid_weights_for_source_target(self, area, items, grid_size, link, affected_radius, affected_value_for_item,
                                           used_value):
        # Area: ((-20.0, -20.0), (444.0, -20.0), (444.0, 268.0), (-20.0, 268.0))
        # todo: area不是从 0,0 起步,但是没关系可以处理
        # 计算区域宽度和高度
        area_min = area[0]
        area_max = area[2]
        width = area[2][0] - area[0][0]
        height = area[2][1] - area[0][1]

        # 计算网格行数和列数
        num_cols = int(width / grid_size)
        num_rows = int(height / grid_size)

        # 创建网格（一个二维数组，初始化为0）
        grid = np.zeros((num_rows, num_cols))
        # 权重矩阵
        weight = [[1] * num_cols for _ in range(num_rows)]

        # 权重建立,还缺少线与线之间的间距，已完成
        for item in items:
            # 创建多边形路径
            poly_path = Path(item.boundary)
            # 创建版图的多边形
            item_polygon = Polygon(item.boundary)
            # 对 boundary 进行扩展（扩展半径可调整，单位同 grid_size）
            affected_polygon = item_polygon.buffer(affected_radius)

            # 遍历网格并检查是否在多边形内
            start_time = datetime.now()
            for row in range(num_rows):
                for col in range(num_cols):
                    grid_x = np.round(area_min[0] + col * grid_size + grid_size / 2, decimals=6)
                    grid_y = np.round(area_max[1] - row * grid_size - grid_size / 2, decimals=6)
                    point = Point(grid_x, grid_y)

                    if poly_path.contains_point((grid_x, grid_y)):
                        grid[row, col] = 1  # 标记占据
                        weight[row][col] = used_value  # 设置为 Used
                    elif affected_polygon.contains(point):
                        weight[row][col] = affected_value_for_item  # 设置为 Affected


            # 针对传进来的source_item和target_item新建其端口，应该是只针对连接的那几个端口设定为2
            # 精度还是有问题，5还是不够，因为有的端口是4为最小差，比如M4的3号端口
            for item_name, port_num in link.items():
                if item.name == item_name:
                    port_index = (int(port_num) - 1)
                    port = item.ports[port_index]

                    poly_path = Path(port['coordinates'])
                    for row in range(num_rows):
                        for col in range(num_cols):
                            grid_x = (area_min[0] + col * grid_size + grid_size / 2)
                            grid_y = (area_max[1] - row * grid_size - grid_size / 2)
                            # if col == 0 and row == 0:
                            #     print("当前坐标： ", grid_x, ", " ,grid_y)
                            #     print("结果：", self.is_point_in_polygon((grid_x, grid_y), item.boundary))
                            if poly_path.contains_point((grid_x, grid_y)):
                                grid[row, col] = 2  # 标记端口

            end_time = datetime.now()
            elapsed_time = end_time - start_time
            print(f"创建grid和weight耗时: {elapsed_time.total_seconds():.2f} 秒")

        # 将网格输出到 txt 文件,暂不需要
        # grid_file = r"D:\IDEA2020\Project\EDA_Last\src\temp\data\case01InputData\grid\grid_output_for_source_target.txt"
        # with open(grid_file, "w") as f:
        #     for row in grid:
        #         f.write(" ".join([f"{int(cell):2d}" for cell in row]) + "\n")

        # 画一下网格点图
        # DataUtils.plot_grid(grid_file)

        # 输出网格大小
        print(f"Grid dimensions: {num_rows}x{num_cols}")

        return grid, weight

    # 已弃用
    def set_grid_weights_for_source_target_custom(self, area, items, grid_size, link, affected_radius,
                                                  affected_value_for_item,
                                                  used_value):
        # Area: ((-20.0, -20.0), (444.0, -20.0), (444.0, 268.0), (-20.0, 268.0))
        # Area can start from non-zero coordinates.
        area_min = area[0]  # (-20.0, -20.0)
        area_max = area[2]
        area_max_coord = area[2]  # (444.0, 268.0) - Represents the top-right corner conceptually
        width = area_max_coord[0] - area_min[0]  # 444.0 - (-20.0) = 464.0
        height = area_max_coord[1] - area_min[1]  # 268.0 - (-20.0) = 288.0

        # --- Grid Calculation Consideration ---
        # Using int() truncates. If width = 464 and grid_size = 10, num_cols = int(46.4) = 46.
        # The grid covers columns 0 to 45. The x-coordinates covered are from
        # area_min[0] up to area_min[0] + 46 * grid_size = -20 + 460 = 440.
        # The region from x=440 to x=444 is not covered by grid centers calculated this way.
        # If full coverage *by grid cells* is essential, use math.ceil:
        # num_cols = math.ceil(width / grid_size)
        # num_rows = math.ceil(height / grid_size)
        # However, using int() is correct if you define the grid size based on how many
        # full cells fit within the width/height starting from area_min.
        # Let's stick to int() as per your original code for now.
        num_cols = math.ceil(width / grid_size)
        num_rows = math.ceil(height / grid_size)

        # Ensure grid dimensions are valid
        if num_rows <= 0 or num_cols <= 0:
            print(f"Warning: Invalid grid dimensions ({num_rows}x{num_cols}) calculated.")
            print(f"Width={width}, Height={height}, GridSize={grid_size}")
            return np.zeros((0, 0)), []  # Return empty structures

        print(f"Calculated Grid dimensions: {num_rows}x{num_cols}")

        # Create grid (a NumPy array, initialized to 0 for type)
        # Use integers for grid markings (0: empty, 1: occupied, 2: port)
        grid = np.zeros((num_rows, num_cols), dtype=int)
        # Weight matrix (list of lists, initialized to default weight 1.0 or similar)
        # Use float for weights
        default_weight = 1.0
        weight = [[default_weight] * num_cols for _ in range(num_rows)]

        # --- Process Items ---
        for item in items:
            item_boundary = item.boundary  # e.g., [[259.0, 20.0], ...]

            # --- Still using Shapely for Buffering ---
            # If you also want to replace this, it requires a geometry library or complex implementation
            # Create版图的多边形 (using Shapely for buffer)
            try:
                item_polygon_shapely = ShapelyPolygon(item_boundary)
                # 对 boundary 进行扩展
                affected_polygon_shapely = item_polygon_shapely.buffer(affected_radius)
            except Exception as e:
                print(f"Warning: Could not create or buffer Shapely polygon for item {item.name}: {e}")
                # Decide how to handle: skip item, use original boundary only?
                item_polygon_shapely = None
                affected_polygon_shapely = None  # Cannot calculate affected area
            # --- End Shapely Buffering ---

            # Iterate through grid cells
            start_time = datetime.now()
            for row in range(num_rows):
                for col in range(num_cols):
                    # Calculate the center coordinates of the current grid cell
                    # Grid origin corresponds to area_min. Row 0 is at the "top" (max y).
                    grid_center_x = area_min[0] + col * grid_size + grid_size / 2
                    # NumPy arrays are (row, col), where row increases downwards.
                    # Grid y coordinates decrease as row index increases.
                    # grid_center_y = area_min[1] + (num_rows - 1 - row) * grid_size + grid_size / 2
                    # Alternative y-calculation based on your original code (using area_max[1]):
                    # grid_center_y = area_max_coord[1] - row * grid_size - grid_size / 2
                    # Let's verify this:
                    # For row 0: area_max_coord[1] - grid_size / 2 (Top row centers) -> Correct
                    # For row num_rows-1: area_max_coord[1] - (num_rows-1)*grid_size - grid_size/2 (Bottom row centers) -> Correct
                    # Both y-calculations should yield similar results if area/height/num_rows are consistent.
                    # Using the original calculation for consistency:
                    grid_center_y = area_max_coord[1] - row * grid_size - grid_size / 2

                    point_coords = (grid_center_x, grid_center_y)
                    point_for_shapely = ShapelyPoint(point_coords)  # Needed for affected_polygon check

                    # --- Check 1: Is the grid cell center inside the item's boundary? ---
                    # Replace Path.contains_point with our custom function
                    if self.is_point_in_polygon(point_coords, item_boundary):
                        grid[row, col] = 1  # Mark as occupied
                        weight[row][col] = used_value  # Set weight to Used

                    # --- Check 2: If not inside item, is it inside the affected (buffered) area? ---
                    # We still use Shapely for the buffered polygon check here.
                    elif affected_polygon_shapely and affected_polygon_shapely.contains(point_for_shapely):
                        # Ensure we don't overwrite a 'used' cell if buffer overlaps significantly
                        if grid[row, col] == 0:  # Only mark if not already marked as occupied
                            weight[row][col] = affected_value_for_item  # Set weight to Affected


            # --- Process Ports for this Item ---
            for item_name, port_num in link.items():
                if item.name == item_name:
                    try:
                        port_index = int(port_num) - 1
                        if 0 <= port_index < len(item.ports):
                            port = item.ports[port_index]
                            port_coordinates = port.get('coordinates')  # Get port boundary

                            if port_coordinates and len(port_coordinates) >= 3:
                                # Iterate through grid cells to find which ones contain the port center
                                for row in range(num_rows):
                                    for col in range(num_cols):
                                        # Recalculate center (or retrieve if stored)
                                        grid_center_x = area_min[0] + col * grid_size + grid_size / 2
                                        grid_center_y = area_max_coord[1] - row * grid_size - grid_size / 2
                                        point_coords = (grid_center_x, grid_center_y)
                                        if col == 0 and row == 0:
                                            print("当前坐标：", point_coords)
                                            print("结果：", self.is_point_in_polygon(point_coords, port_coordinates))


                                        # --- Check if grid cell center is inside the port boundary ---
                                        # Replace Path.contains_point with our custom function
                                        if self.is_point_in_polygon(point_coords, port_coordinates):
                                            grid[row, col] = 2  # Mark as Port
                        else:
                            print(
                                f"Warning: Invalid port index {port_index} for item {item.name} with {len(item.ports)} ports.")
                    except (ValueError, IndexError, KeyError, TypeError) as e:
                        print(f"Warning: Error processing port {port_num} for item {item.name}: {e}")

        end_time = datetime.now()
        elapsed_time = end_time - start_time
        print(f"创建grid和weight耗时: {elapsed_time.total_seconds():.2f} 秒")

        return grid, weight

    def set_grid_weights_for_source_target_custom_corners(self, area, items, grid_size, link, affected_radius,
                                                          affected_value_for_item,
                                                          used_value, Flag):
        # 区域和网格计算部分保持不变
        area_min = area[0]
        area_max_coord = area[2]
        width = area_max_coord[0] - area_min[0]
        height = area_max_coord[1] - area_min[1]

        num_cols = math.ceil(width / grid_size)
        num_rows = math.ceil(height / grid_size)

        if num_rows <= 0 or num_cols <= 0:
            print(f"警告: 计算出的网格维度无效 ({num_rows}x{num_cols})。")
            print(f"宽度={width}, 高度={height}, 网格尺寸={grid_size}")
            return np.zeros((0, 0)), []

        # print(f"计算出的网格维度: {num_rows}x{num_cols}")

        grid = np.zeros((num_rows, num_cols), dtype=int)
        default_weight = 1.0
        weight = [[default_weight] * num_cols for _ in range(num_rows)]

        grid_2 = copy.deepcopy(grid)
        weight_2 = copy.deepcopy(weight)

        if  Flag == 13:
            return grid_2, weight_2

        # --- 处理 Items ---
        for item in items:
            item_boundary = item.boundary

            # --- 仍然使用 Shapely 进行缓冲处理 ---
            try:
                item_polygon_shapely = ShapelyPolygon(item_boundary)
                affected_polygon_shapely = item_polygon_shapely.buffer(affected_radius)
            except Exception as e:
                print(f"警告: 无法为 item {item.name} 创建或缓冲 Shapely 多边形: {e}")
                item_polygon_shapely = None
                affected_polygon_shapely = None
            # --- 结束 Shapely 缓冲处理 ---

            # --- 主体区域检查（仍然检查中心点，您可以根据需要也改成检查角点）---
            minx, miny = 10000000000, 10000000000
            maxx, maxy = -10000000000, -10000000000
            # print("port_coordinates  ", port_coordinates)

            for x, y in item.boundary:
                temp_y = height - y

                if x < minx:
                    minx = x
                if temp_y < miny:
                    miny = temp_y
                if x > maxx:
                    maxx = x
                if temp_y > maxy:
                    maxy = temp_y

            i0, j0, i1, j1 = int(minx / grid_size), int(miny / grid_size), int(maxx / grid_size), int(maxy / grid_size)



            for row in range(j0, j1 + 1):
                for col in range(i0, i1 + 1):
                    grid_center_x = area_min[0] + col * grid_size + grid_size / 2
                    grid_center_y = area_max_coord[1] - row * grid_size - grid_size / 2
                    point_coords = (grid_center_x, grid_center_y)
                    point_for_shapely = ShapelyPoint(point_coords)

                    # 检查1: 中心点是否在 item 边界内?
                    # 注意：这里仍然用的是中心点检查 item 区域，如果需要也可以改成角点检查
                    if self.is_point_in_polygon(point_coords, item_boundary):
                        grid[row, col] = 1
                        weight[row][col] = used_value
                    # 检查2: 如果不在 item 内，是否在受影响区域内?
                    elif affected_polygon_shapely and affected_polygon_shapely.contains(point_for_shapely):
                        if grid[row, col] == 0:
                            weight[row][col] = affected_value_for_item

            # ******** 修改开始：将中心点检查改为角点检查 ********
            for item_name, port_num in link.items():
                if item.name == item_name:

                    port_index = int(port_num) - 1
                    if 0 <= port_index < len(item.ports):
                        port = item.ports[port_index]
                        port_coordinates = port.get('coordinates')  # 获取端口边界坐标

                        minx, miny = 10000000000,10000000000
                        maxx, maxy = -10000000000, -10000000000
                        # print("port_coordinates  ", port_coordinates)

                        for x,y in port_coordinates:
                            temp_y = height - y

                            if x < minx:
                                minx = x
                            if temp_y < miny:
                                miny = temp_y
                            if x > maxx:
                                maxx = x
                            if temp_y > maxy:
                                maxy = temp_y

                        i0, j0, i1, j1 = int(minx / grid_size), int(miny / grid_size), int(maxx / grid_size), int(maxy / grid_size)
                        for row in range(j0, j1+1):
                            for col in range(i0, i1+1):
                                grid[row, col] = 2

                    else:
                        print(f"警告: item {item.name} 的端口索引 {port_index} 无效 (总端口数: {len(item.ports)})。")

            # ******** 修改结束 *******

        # 可选的文件输出和绘图部分保持不变...
        return grid, weight # 对应13层

    # def set_grid_2_for_port(self, area, items, grid, grid_size, link):
    #     area_min = area[0]
    #     area_max_coord = area[2]
    #     width = area_max_coord[0] - area_min[0]
    #     height = area_max_coord[1] - area_min[1]
    #
    #     num_cols = int(width / grid_size)
    #     num_rows = int(height / grid_size)
    #
    #     if num_rows <= 0 or num_cols <= 0:
    #         print(f"警告: 计算出的网格维度无效 ({num_rows}x{num_cols})。")
    #         print(f"宽度={width}, 高度={height}, 网格尺寸={grid_size}")
    #         return np.zeros((0, 0)), []
    #
    #     print(f"计算出的网格维度: {num_rows}x{num_cols}")
    #
    #
    #     # --- 处理 Items ---
    #     for item in items:
    #         # 先判断这个item是不是link所包含的item
    #         if item.name not in link:
    #             continue
    #
    #         item_boundary = item.boundary
    #
    #         # --- 仍然使用 Shapely 进行缓冲处理 ---
    #         try:
    #             item_polygon_shapely = ShapelyPolygon(item_boundary)
    #             # affected_polygon_shapely = item_polygon_shapely.buffer(affected_radius)
    #         except Exception as e:
    #             print(f"警告: 无法为 item {item.name} 创建或缓冲 Shapely 多边形: {e}")
    #             item_polygon_shapely = None
    #             # affected_polygon_shapely = None
    #         # --- 结束 Shapely 缓冲处理 ---
    #
    #         # --- 在这里处理该 Item 的端口 (Ports) ---
    #         # ******** 修改开始：将中心点检查改为角点检查 ********
    #         for item_name, port_num in link.items():
    #             if item.name == item_name:
    #                 try:
    #                     port_index = int(port_num) - 1
    #                     if 0 <= port_index < len(item.ports):
    #                         port = item.ports[port_index]
    #                         port_coordinates = port.get('coordinates')  # 获取端口边界坐标
    #
    #                         if port_coordinates and len(port_coordinates) >= 3:
    #                             # 遍历网格单元格
    #                             for row in range(num_rows):
    #                                 for col in range(num_cols):
    #                                     # 如果该单元格已被标记为 Item 外部 (0)，则跳过端口检查
    #                                     if grid[row, col] == 0:
    #                                         continue
    #
    #                                     # --- 计算当前单元格的四个角点坐标 ---
    #                                     cell_min_x = area_min[0] + col * grid_size
    #                                     cell_max_x = cell_min_x + grid_size
    #                                     # Y坐标：row 0 对应 area_max_coord[1]
    #                                     cell_max_y = area_max_coord[1] - row * grid_size  # 单元格上边界Y
    #                                     cell_min_y = cell_max_y - grid_size  # 单元格下边界Y
    #
    #                                     corners = [
    #                                         (cell_min_x, cell_min_y),  # 左下角
    #                                         (cell_max_x, cell_min_y),  # 右下角
    #                                         (cell_max_x, cell_max_y),  # 右上角
    #                                         (cell_min_x, cell_max_y)  # 左上角
    #                                     ]
    #
    #                                     # --- 检查是否有任何一个角点落在端口多边形内 ---
    #                                     is_port_cell = False
    #                                     for corner_point in corners:
    #                                         # 注意：这里需要调用您实现的 is_point_in_polygon
    #                                         if self.is_point_in_polygon(corner_point, port_coordinates):
    #                                             is_port_cell = True
    #                                             break  # 只要有一个角点在内部，就确认是端口单元格，停止检查其他角点
    #
    #                                     # 如果该单元格的任何一个角点在端口内
    #                                     if is_port_cell:
    #                                         # 即使它之前可能被标记为受影响区域，现在也标记为端口
    #                                         grid[row, col] = 2  # 标记为端口
    #                                         # 可以选择性地为端口设置特定权重
    #                                         # weight[row][col] = port_weight
    #
    #                     else:
    #                         print(f"警告: item {item.name} 的端口索引 {port_index} 无效 (总端口数: {len(item.ports)})。")
    #                 except (ValueError, IndexError, KeyError, TypeError) as e:
    #                     print(f"警告: 处理 item {item.name} 的端口 {port_num} 时出错: {e}")
    #         # ******** 修改结束 *******
    #
    #     # 可选的文件输出和绘图部分保持不变...
    #
    #     return grid


    def is_point_in_polygon(self, point, polygon_vertices):
        """
        Checks if a point is inside a polygon using the Ray Casting algorithm.

        Args:
            point (tuple): A tuple (x, y) representing the point coordinates.
            polygon_vertices (list): A list of tuples [(x1, y1), (x2, y2), ...]
                                     representing the vertices of the polygon in order.
                                     The last vertex is assumed connected to the first.

        Returns:
            bool: True if the point is inside the polygon, False otherwise.
        """
        px, py = point
        num_vertices = len(polygon_vertices)
        is_inside = False

        # The polygon needs at least 3 vertices
        if num_vertices < 3:
            return False

        p1x, p1y = polygon_vertices[0]
        for i in range(num_vertices + 1):
            p2x, p2y = polygon_vertices[i % num_vertices]  # Wrap around using modulo

            # Check if the point's y-coordinate is between the edge's y-coordinates
            if min(p1y, p2y) <= py < max(p1y, p2y):
                # Avoid division by zero for vertical lines
                if p2y - p1y != 0:
                    # Calculate the x-coordinate of the intersection of the ray and the edge's line
                    x_intersection = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    # If the intersection x-coordinate is to the right of the point, count it
                    if x_intersection >= px:  # Changed from > to >= to include points ON the boundary ray intersection
                        is_inside = not is_inside
            # Special case: point is collinear with a horizontal segment
            elif p1y == p2y == py:
                if min(p1x, p2x) <= px <= max(p1x, p2x):
                    return True  # Point lies on a horizontal edge, consider it 'inside' or on boundary

            # Move to the next edge
            p1x, p1y = p2x, p2y

        # Handle point exactly on a vertex (optional, ray casting might miss this sometimes)
        # A more robust check might be needed if exact vertex inclusion is critical.
        # The current logic might include points on some boundaries but not others based on ray crossing.
        # If you need strict "inside only" or "inside including boundary", adjustments might be needed.
        for i in range(num_vertices):
            if polygon_vertices[i] == point:
                return True  # Point is exactly on a vertex

        return is_inside

    def multi_source_region_search_V2(self, grid, weights, source_area_Out, target_area_Out, start_time, max_time):
        source_area_in = source_area_Out.copy()
        target_area_in = target_area_Out.copy()
        rows, cols = len(grid), len(grid[0])

        # 定义移动方向（上下左右）
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # 函数：计算从一个点到目标区域的最短路径
        def find_path(target, source_area, parent):
            x, y = target
            path = []
            while (x, y) is not None:
                path.append((x, y))
                if [x, y] in source_area:
                    break
                x, y = parent[x][y]
            path.reverse()
            return path

        def dijkstra_search(source_area, target_area, start_time, max_time):
            dist = [[float('inf')] * cols for _ in range(rows)]  # 存储最短路径
            parent = [[None] * cols for _ in range(rows)]  # 用于回溯路径
            pq = []  # 优先队列，用于Dijkstra算法
            # 初始化源区域
            for x, y in source_area:
                dist[x][y] = 0
                heapq.heappush(pq, (0, x, y))  # (距离, x, y)

            # 最短路径搜索
            while pq:
                # ==================== 新增代码: 第3步 ====================
                # 在每次从队列取元素时检查时间，这是最高频的操作，控制最精确
                if start_time and max_time and (time.time() - start_time >= max_time):
                    # 如果超时，直接抛出我们自定义的异常
                    raise TimeoutException("Dijkstra 寻路超时！")
                # =========================================================

                current_dist, x, y = heapq.heappop(pq)

                # 如果当前节点是目标区域的一部分，则停止
                if [x, y] in target_area:
                    return (x, y), parent  # 返回找到的目标节点

                # 如果当前节点的距离大于已知最短距离，跳过
                if current_dist > dist[x][y]:
                    continue

                # 遍历邻居节点
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    # if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:  # 检查有效邻居
                    if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != 1:  # 检查有效邻居
                        new_dist = dist[x][y] + weights[nx][ny]
                        if new_dist < dist[nx][ny]:  # 更新距离
                            dist[nx][ny] = new_dist
                            parent[nx][ny] = (x, y)  # 记录父节点
                            heapq.heappush(pq, (new_dist, nx, ny))
            return None, None  # 如果没有找到目标，返回两个None

        # 存储路径
        all_paths = []
        all_connected = []
        used_source = []
        used_target = []
        # 步骤 1：首先找到最接近的两个源点连接起来
        while len(source_area_in) >= 1:
            # 跑一次dijkstra
            nearest_target, parent = dijkstra_search(source_area_in, target_area_in, start_time, max_time)
            if (nearest_target, parent) == (None, None):
                return grid, weights, all_paths
            if nearest_target:
                path = find_path(nearest_target, source_area_in, parent)

                used_source.append(path[0])
                used_target.append(nearest_target)
                # 更新起点终点
                source_area_in.remove(list(path[0]))
                target_area_in.remove(list(nearest_target))
                newpath = [list(sublist) for sublist in path]
                target_area_in.extend(newpath)

                all_paths.append(path)
                all_connected.append(nearest_target)
        return grid, weights, all_paths

    def getSourceTargetFor(self, items, area, links, grid_size, affected_radius, affected_value_for_item, used_value,
                        port_boundary_index, Flag):
        area_min = area[0]
        area_max = area[2]
        # 需要对每一条link都生成一个grid，存到grid_list返回回去，已完成
        grid_list = []
        weight_list = []
        # 存储每个连接的端口信息
        connected_ports = []

        # 遍历 Links 中的每个连接
        # print("===links===", links)
        # for item in items:
        #     print("===item===", item.to_string())
        for link in links:
            link_ports = []

            grid, weight = self.set_grid_weights_for_source_target_custom_corners(area, items, grid_size, link,
                                                                                      affected_radius,
                                                                                      affected_value_for_item,
                                                                                      used_value, Flag)
            grid_list.append(grid)
            weight_list.append(weight)



            # print(f"每一次link里面grid, weight: {elapsed_time.total_seconds():.2f} 秒")

            # 遍历链接中的每个模块与端口

            for module_name, port_index in link.items():
                # 查找模块
                # module = next(item for item in items if item.name == module_name)
                module = next((item for item in items if item.name.strip() == module_name.strip()), None)
                # 获取对应的端口（port_index 是字符串形式，需要转为整数）
                port_index = int(port_index) - 1  # 转换为从 0 开始的索引
                # 获取该端口的坐标和类型
                port_info = {
                    'module': module_name,
                    'port_coordinates': module.ports[port_index]['coordinates'],
                    'port_layer': module.ports[port_index]['type'],
                    'is_third_port': True if port_index == 2 else False
                }
                link_ports.append(port_info)
            # 将每个链接的端口信息存入列表
            connected_ports.append(link_ports)

        # 到此还只是拿到各个端口的连接情况
        # 需要将其转化成source和target，这应该如何选择？
        # 暂时用每个端口的顶点作为出入口
        # 存储所有随机选取的源点
        source_list = []
        target_list = []
        modulelist_Target = []
        modulelist_Source = []
        is_third_port_S = []
        is_third_port_T = []
        # 遍历每个连接的端口信息
        for link_ports in connected_ports:
            flag = 0
            temp_list_S = []
            temp_list_T = []
            temp_third_port_S = []
            temp_third_port_T = []
            temp_modulelist_S = []
            temp_modulelist_T = []
            for port in link_ports:
                # 获取端口的坐标（即多边形的顶点）
                coordinates = port['port_coordinates']
                if flag == 0:
                    temp_list_T.append(list(coordinates[port_boundary_index]))
                    temp_third_port_T.append(port['is_third_port'])

                    temp_modulelist_T.append(port['module'])
                    flag = 1
                else:
                    temp_list_S.append(list(coordinates[port_boundary_index]))
                    temp_third_port_S.append(port['is_third_port'])

                    temp_modulelist_S.append(port['module'])

            source_list.append(temp_list_S)
            target_list.append(temp_list_T)
            is_third_port_S.append(temp_third_port_S)
            is_third_port_T.append(temp_third_port_T)
            modulelist_Target.append(temp_modulelist_T)
            modulelist_Source.append(temp_modulelist_S)

        source_list_grid = []
        target_list_grid = []
        # 处理 source_list 中的点
        # 处理的不对应,不对应是因为精度有误差，有的时候会被放到其他格子里面，能不能做一个四周的探索非1的格子然后赋值，已完成
        for i in range(len(source_list)):
            source_group = source_list[i]
            index = 0
            grid = grid_list[i]
            source_group_grid = []
            for point in source_group:
                x, y = point
                grid_x, grid_y = self.convert_to_grid_coordinates(x, y, area_min, area_max, grid_size)
                # 需要做一下探索，已完成
                if grid[grid_x][grid_y] == 2:
                    source_group_grid.append([grid_x, grid_y])
                    # dict_source.update({[grid_x, grid_y] : list_source_index[index]})
                    # index += 1
                    continue
                else:
                    # 定义偏移量列表，表示邻接方向
                    offsets = [
                        (1, 0),  # 右
                        (0, 1),  # 上
                        (-1, 0),  # 左
                        (0, -1),  # 下
                        (1, 1),  # 右上
                        (-1, 1),  # 左上
                        (-1, -1),  # 左下
                        (1, -1),  # 右下
                        (2, 0),  # 右
                        (0, 2),  # 上
                        (-2, 0),  # 左
                        (0, -2),  # 下
                    ]

                    # 遍历邻接方向
                    for dx, dy in offsets:
                        nx, ny = grid_x + dx, grid_y + dy
                        # 检查是否越界
                        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                            # 如果该点为2
                            if grid[nx][ny] == 2:
                                source_group_grid.append([nx, ny])
                                # dict_source.update({[nx, ny]: list_source_index[index]})
                                # index += 1
                                break  # 找到一个满足条件的点后退出循环
            source_list_grid.append(source_group_grid)

        # 处理 target_list 中的点
        for i in range(len(target_list)):
            target_group = target_list[i]
            grid = grid_list[i]
            target_group_grid = []
            for point in target_group:
                x, y = point
                grid_x, grid_y = self.convert_to_grid_coordinates(x, y, area_min, area_max, grid_size)

                # 探索
                if grid[grid_x][grid_y] == 2:
                    target_group_grid.append([grid_x, grid_y])
                    continue
                else:
                    # 定义偏移量列表，表示邻接方向
                    # todo: 这个探索策略有些时候不够用，可能离得远比如左上两格
                    offsets = [
                        (1, 0),  # 右
                        (0, 1),  # 上
                        (-1, 0),  # 左
                        (0, -1),  # 下
                        (1, 1),  # 右上
                        (-1, 1),  # 左上
                        (-1, -1),  # 左下
                        (1, -1),  # 右下
                        (2, 0),  # 右
                        (0, 2),  # 上
                        (-2, 0),  # 左
                        (0, -2),  # 下
                        (2, 1),  # 右
                        (2, -1),  # 上
                        (-2, -1),  # 左
                        (-2, 1),  # 下
                    ]

                    # 遍历邻接方向
                    for dx, dy in offsets:
                        nx, ny = grid_x + dx, grid_y + dy
                        # 检查是否越界
                        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                            # 如果该点为2
                            if grid[nx][ny] == 2:
                                target_group_grid.append([nx, ny])
                                break  # 找到一个满足条件的点后退出循环
            target_list_grid.append(target_group_grid)

        # 插入一个处理逻辑：我的source点或者target点若是link中3号端口的点，就换位置到一个合法位置，并记录这个移动的路径，不记录好像也行
        # 将端口3的那个选点挪动到合法位置
        move_list = []
        for i in range(len(source_list_grid)):
            grid = grid_list[i]
            source_group = source_list_grid[i]
            is_third_port_S_group = is_third_port_S[i]
            temp_remove_item = []
            temp_add_item = []
            for i in range(len(source_group)):
                source_item = source_group[i]
                grid_x, grid_y = source_item
                # 如果是3号端口的点
                if is_third_port_S_group[i]:
                    # 探索后挪到合法位置
                    # 定义偏移量列表，表示邻接方向
                    offsets = [
                        (2, 0),  # 右
                        (0, 2),  # 上
                        (-2, 0),  # 左
                        (0, -2),  # 下
                        (3, 0),  # 右
                        (0, 3),  # 上
                        (-3, 0),  # 左
                        (0, -3),  # 下
                    ]

                    # 遍历邻接方向
                    for dx, dy in offsets:
                        nx, ny = grid_x + dx, grid_y + dy
                        # 检查是否越界
                        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                            # 如果该点为0
                            if grid[nx][ny] == 0:
                                # todo: 后续有需要可以再这里记录端口3的出入点的合法挪动路径
                                temp_remove_item.append(source_item)
                                temp_add_item.append([nx, ny])
                                break  # 找到一个满足条件的点后退出循环
            for i in range(len(temp_remove_item)):
                source_group.remove(temp_remove_item[i])
                source_group.append(temp_add_item[i])

        for i in range(len(target_list_grid)):
            grid = grid_list[i]
            target_group = target_list_grid[i]
            is_third_port_T_group = is_third_port_T[i]
            temp_remove_item = []
            temp_add_item = []
            for i in range(len(target_group)):
                target_item = target_group[i]
                grid_x, grid_y = target_item
                # 如果是3号端口的点
                if is_third_port_T_group[i]:
                    # 探索后挪到合法位置
                    # 定义偏移量列表，表示邻接方向
                    offsets = [
                        (2, 0),  # 右
                        (0, 2),  # 上
                        (-2, 0),  # 左
                        (0, -2),  # 下
                        (3, 0),  # 右
                        (0, 3),  # 上
                        (-3, 0),  # 左
                        (0, -3),  # 下
                    ]

                    # 遍历邻接方向
                    for dx, dy in offsets:
                        nx, ny = grid_x + dx, grid_y + dy
                        # 检查是否越界
                        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                            # 如果该点为0
                            if grid[nx][ny] == 0:
                                # todo: 后续有需要可以再这里记录端口3的出入点的合法挪动路径
                                temp_remove_item.append(target_item)
                                temp_add_item.append([nx, ny])
                                break  # 找到一个满足条件的点后退出循环
            for i in range(len(temp_remove_item)):
                target_group.remove(temp_remove_item[i])
                target_group.append(temp_add_item[i])

        return source_list, target_list, source_list_grid, target_list_grid, grid_list, weight_list, modulelist_Target, modulelist_Source, connected_ports




    def update_grid(self, grid, paths):
        for path in paths:
            for x, y in path:
                grid[x][y] = 1  # 将路径节点标记为障碍

        # 将网格输出到 txt 文件,暂不需要
        # grid_file = r"D:\IDEA2020\Project\EDA_Last\src\temp\data\case01InputData\grid\grid_output11111.txt"
        # with open(grid_file, "w") as f:
        #     for row in grid:
        #         f.write(" ".join([f"{int(cell):2d}" for cell in row]) + "\n")

        return grid

    def update_weight(self, weight, paths, grid_size, affected_radius, affected_value_for_path, used_value):
        # 取两倍grid_size
        affected_range = (int)(affected_radius / grid_size)

        for path in paths:
            for x, y in path:
                weight[x][y] = used_value  # 将路径节点标记为障碍
                # 遍历扩展范围内的格子
                for dx in range(-affected_range, affected_range + 1):
                    for dy in range(-affected_range, affected_range + 1):
                        nx, ny = x + dx, y + dy  # 计算新坐标

                        # 确保扩展格子在网格范围内
                        if 0 <= nx < len(weight) and 0 <= ny < len(weight[0]):
                            # 如果当前格子不是 "Used"，标记为 "Affected"
                            if weight[nx][ny] != used_value:
                                weight[nx][ny] = affected_value_for_path
        return weight


# --- DataUtils 类 ---
# (此类中的所有函数都已保留，代码与上一版本相同)
class DataUtils():
    @staticmethod
    def get_max_item_number_from_links_regex(links_list):
        """
        使用正则表达式从 links 列表中提取所有形如 'M<数字>' 的键，并返回最大的数字。
        """
        max_number = 0
        if not links_list:
            return 0

        # 正则表达式：匹配以 'M' 开头，后跟一个或多个数字的模式
        # \d+ 表示一个或多个数字
        # ( ) 用于捕获数字部分
        pattern = re.compile(r'^M(\d+)$')

        for link_dict in links_list:
            for key in link_dict.keys():
                match = pattern.match(key)
                if match:
                    number_str = match.group(1)  # group(1) 获取第一个捕获组的内容 (即数字部分)
                    number = int(number_str)
                    if number > max_number:
                        max_number = number
        return max_number
    @staticmethod
    def filter_invalid_links(links_list, items_list):
        """
        从 links_list 中移除那些包含在 items_list 中不存在的 item 名称的字典。

        参数:
        links_list (list): 一个字典列表，例如 [{'M6': '2', 'M12': '2'}, ...]
        items_list (list): Item 对象的列表，每个对象都有一个 'name' 属性。

        返回:
        list: 一个新的列表，其中只包含所有键都存在于 items_list 中的字典。
        """
        # 1. 从 items_list 中提取所有有效的 item 名称，并存入一个集合以便快速查找
        valid_item_names = set()
        for item_obj in items_list:
            valid_item_names.add(item_obj.name)

        # 2. 遍历 links_list，筛选出有效的 link 字典
        filtered_links = []
        for link_dict in links_list:
            all_keys_valid = True  # 假设当前字典中的所有键都是有效的
            for key_name in link_dict.keys():
                if key_name not in valid_item_names:
                    all_keys_valid = False  # 发现一个无效的键
                    break  # 不需要再检查这个字典的其他键了

            if all_keys_valid:
                filtered_links.append(link_dict)

        return filtered_links
    @staticmethod
    def calculate_overall_bounding_box_area(connected_items, list_of_original_path_groups) -> float:
        """
        计算所有 connected_items 和路径列表所占据的整体外包矩形的面积。

        该方法通过找到所有几何形状中的最小和最大X、Y坐标来确定一个能包围所有形状的
        最小矩形，然后计算这个矩形的面积。

        请注意：这个面积不是各个形状的面积之和，也不是它们几何并集（考虑重叠）的面积。
        它通常会大于实际形状覆盖的区域，因为它包含了形状之间的空白区域。

        Args:
            connected_items (List[Any]): "连接的" item 列表。每个 item 应能提供一个
                                         'boundary' 属性/键, 其值为点列表 [[x,y], ...]。
            list_of_original_path_groups (ListOfOriginalPathGroups): 原始坐标的路径组列表。
                结构为: [ path_group_1, ... ]
                其中 path_group_1 = [ segment_1, ... ]
                而 segment_1 = [ [x,y]_1, ... ]

        Returns:
            float: 计算得到的外包矩形面积。如果没有找到有效的点，则返回 0.0。
        """

        min_x_overall = math.inf
        min_y_overall = math.inf
        max_x_overall = -math.inf
        max_y_overall = -math.inf

        points_found = False

        # 1. 处理 connected_items 中的点
        for item_idx, item in enumerate(connected_items):
            boundary = []  # 初始化为空列表以防item没有有效boundary

            # 尝试获取 boundary 数据 (兼容对象属性和字典键)
            if hasattr(item, 'boundary'):
                item_boundary_attr = getattr(item, 'boundary')
                if isinstance(item_boundary_attr, list):
                    boundary = item_boundary_attr
            elif isinstance(item, dict) and 'boundary' in item:
                item_boundary_dict_val = item['boundary']
                if isinstance(item_boundary_dict_val, list):
                    boundary = item_boundary_dict_val
            # else:
            # 可以选择性地为没有有效 boundary 的 item 添加警告
            # print(f"警告: connected_items 中索引 {item_idx} 的 item 没有有效的 'boundary'。")

            if not boundary:  # 如果 boundary 是空列表或未成功获取
                continue

            for point_idx, point in enumerate(boundary):
                if isinstance(point, (list, tuple)) and len(point) == 2:
                    try:
                        x, y = float(point[0]), float(point[1])
                        min_x_overall = min(min_x_overall, x)
                        min_y_overall = min(min_y_overall, y)
                        max_x_overall = max(max_x_overall, x)
                        max_y_overall = max(max_y_overall, y)
                        points_found = True
                    except (ValueError, TypeError):
                        # print(f"警告: connected_items[{item_idx}] 的 boundary 点 {point_idx} '{point}' 格式无效。")
                        pass  # 跳过格式错误的点
                # else:
                # print(f"警告: connected_items[{item_idx}] 的 boundary 点 {point_idx} '{point}' 不是有效的坐标对。")

        # 2. 处理 list_of_original_path_groups 中的点
        if isinstance(list_of_original_path_groups, list):
            for group_idx, path_group in enumerate(list_of_original_path_groups):
                if isinstance(path_group, list):
                    for seg_idx, segment in enumerate(path_group):
                        if isinstance(segment, list):
                            for point_idx, point in enumerate(segment):
                                if isinstance(point, (list, tuple)) and len(point) == 2:
                                    try:
                                        x, y = float(point[0]), float(point[1])
                                        min_x_overall = min(min_x_overall, x)
                                        min_y_overall = min(min_y_overall, y)
                                        max_x_overall = max(max_x_overall, x)
                                        max_y_overall = max(max_y_overall, y)
                                        points_found = True
                                    except (ValueError, TypeError):
                                        # print(f"警告: 路径组[{group_idx}]-段[{seg_idx}]-点[{point_idx}] '{point}' 格式无效。")
                                        pass  # 跳过格式错误的点
                                # else:
                                # print(f"警告: 路径组[{group_idx}]-段[{seg_idx}]-点[{point_idx}] '{point}' 不是有效的坐标对。")

        if not points_found:
            print("信息: 未找到任何有效坐标点，无法计算外包矩形面积。")
            return 0.0

        # 计算外包矩形的宽度和高度
        width = max_x_overall - min_x_overall
        height = max_y_overall - min_y_overall

        # 确保宽度和高度有效 (例如，所有点在一条线上时，宽度或高度可能为0)
        # 如果 min/max 更新正确，width/height 不应为负
        if width < 0 or height < 0:
            # 这种情况理论上不应发生，如果 points_found 为 True 且 min/max 逻辑正确
            print(f"警告: 计算出的宽度或高度为负 (宽:{width}, 高:{height})。可能所有点共线或数据问题。返回0面积。")
            return 0.0

        return width * height


    @staticmethod
    def _convert_point_from_grid(
            grid_row,
            grid_col,
            area_min_tuple,  # 例如: (0.0, 0.0)
            area_max_tuple,  # 例如: (546.0, 320.0)
            grid_size
    ):
        """
        (辅助函数) 将单个网格坐标（行，列）转换回原始物理坐标。
        使用 area_min_tuple[0] 作为 min_x，使用 area_max_tuple[1] 作为 max_y (用于Y轴翻转)。
        """
        min_x = area_min_tuple[0]
        max_y_for_inversion = area_max_tuple[1]  # Y轴坐标转换的参考最大Y值

        original_x = float(grid_col) * grid_size + min_x
        original_y = max_y_for_inversion - float(grid_row) * grid_size
        return [original_x, original_y]

    @staticmethod
    def convert_paths_list_from_grid(
            list_of_grid_path_groups,
            area_min,
            area_max,
            grid_size,
    ):
        """
        将一个网格路径的列表（每个路径是点的列表）转换回原始物理坐标路径的列表。

        Args:
            grid_paths_list (ListOfGridPathsInput): 网格路径的列表。
                每个元素是一条路径，该路径是 (grid_row, grid_col) 点的列表/元组的列表。
                例如: [
                        [(r1,c1), (r2,c2), ...],  # path 1
                        [[rA,cA], [rB,cB], ...]  # path 2 (点也可以是列表形式)
                      ]
            area_min (AreaTuple): 原始区域的最小坐标元组 (min_x, min_y)。
            area_max (AreaTuple): 原始区域的最大坐标元组 (max_x, max_y)。
            grid_size (float): 网格单元的尺寸（以原始单位计）。

        Returns:
            ListOfOriginalPaths: 转换后的原始坐标路径的列表。
                结构与输入相同: [[ [x1,y1], [x2,y2], ...], [[xA,yA], ...]]
        """
        all_original_path_groups = []
        if not list_of_grid_path_groups:
            return []

        for i, grid_path_group in enumerate(list_of_grid_path_groups):  # 遍历每个 "路径组"
            current_original_path_group = []
            if not isinstance(grid_path_group, list):
                print(
                    f"警告: 最外层列表(路径组列表)中索引 {i} 的元素 '{grid_path_group}' 不是列表类型，将视为空路径组处理。")
                all_original_path_groups.append(current_original_path_group)
                continue
            for j, grid_segment in enumerate(grid_path_group):  # 遍历路径组中的每个 "路径段"
                current_original_segment = []
                if not isinstance(grid_segment, list):
                    print(
                        f"警告: 路径组索引 {i} 中，段索引 {j} 的元素 '{grid_segment}' 不是列表类型（段），将视为空段处理。")
                    current_original_path_group.append(current_original_segment)
                    continue

                if not grid_segment:  # 如果路径段本身是个空列表
                    current_original_path_group.append(current_original_segment)
                    continue

                for k, grid_point in enumerate(grid_segment):  # 遍历路径段中的每个 "点"
                    if not (isinstance(grid_point, (list, tuple)) and len(grid_point) == 2):
                        print(f"警告: 路径组索引 {i}，段索引 {j}，点索引 {k} 处的网格点 '{grid_point}' 格式无效，已跳过。")
                        continue

                    try:
                        grid_row = int(grid_point[0])
                        grid_col = int(grid_point[1])
                    except (ValueError, TypeError) as e:
                        print(
                            f"警告: 路径组索引 {i}，段索引 {j}，点索引 {k} 处的网格点 '{grid_point}' 无法转换为整数坐标 ({e})，已跳过。")
                        continue

                    original_coords = DataUtils._convert_point_from_grid(
                        grid_row, grid_col, area_min, area_max, grid_size
                    )
                    current_original_segment.append(original_coords)

                current_original_path_group.append(current_original_segment)

            all_original_path_groups.append(current_original_path_group)

        return all_original_path_groups

    @staticmethod
    def _calculate_polygon_area_shoelace(vertices: list) -> float:
        """
        使用鞋带公式计算多边形的面积。

        Args:
            vertices (list): 多边形顶点的列表，每个顶点是 [x, y] 坐标对。
                             例如：[[x1, y1], [x2, y2], ..., [xn, yn]]

        Returns:
            float: 多边形的面积。如果顶点少于3个或输入无效，则返回0.0。
        """
        if not vertices or not isinstance(vertices, list):
            # print("警告: 提供的顶点列表无效或为空。")
            return 0.0

        n = len(vertices)
        if n < 3:
            # print(f"警告: 至少需要3个顶点来构成多边形，当前只有 {n} 个。")
            return 0.0

        # 验证所有顶点是否都是有效的坐标对
        for i, point in enumerate(vertices):
            if not (isinstance(point, list) and len(point) == 2 and
                    all(isinstance(coord, (int, float)) for coord in point)):
                # print(f"警告: 顶点列表中的索引 {i} 处的点 '{point}' 格式无效。面积计为0。")
                return 0.0  # 如果任何点无效，则整个多边形面积为0

        area_sum = 0.0
        for i in range(n):
            x_i = vertices[i][0]
            y_i = vertices[i][1]

            # 获取下一个顶点，对于最后一个顶点，下一个是第一个顶点 (实现环绕)
            x_i_plus_1 = vertices[(i + 1) % n][0]
            y_i_plus_1 = vertices[(i + 1) % n][1]

            area_sum += (x_i * y_i_plus_1) - (x_i_plus_1 * y_i)

        return abs(area_sum) / 2.0

    @staticmethod
    def calculate_item_areas(items: list) -> list:
        """
        计算列表中每个 item 的面积，面积根据其 'boundary' 属性确定。

        Args:
            items (list): 一个 item 对象的列表。
                          每个 item 应该有一个 'boundary' 属性（如果 item 是对象）
                          或 'boundary' 键（如果 item 是字典）。
                          'boundary' 的值应该是一个顶点列表 [[x1,y1], [x2,y2], ...]。

        Returns:
            list: 一个浮点数列表，包含按输入顺序排列的每个 item 的面积。
                  如果某个 item 没有有效的 'boundary'，则其对应面积为 0.0。
        """
        areas_list = []
        if not isinstance(items, list):
            print("警告: 输入的 'items' 不是一个列表。")
            return []

        for idx, item in enumerate(items):
            boundary_points = None
            item_description = f"索引 {idx} 处的项目"  # 用于错误信息

            if hasattr(item, 'boundary'):  # 适用于对象
                boundary_points = item.boundary
                if hasattr(item, 'name'):  # 尝试获取名称以提供更清晰的日志
                    item_description = f"项目 '{item.name}' (索引 {idx})"
                else:
                    item_description = f"项目 (类型: {type(item)}, 索引 {idx})"

            elif isinstance(item, dict) and 'boundary' in item:  # 适用于字典
                boundary_points = item['boundary']
                if 'name' in item:
                    item_description = f"项目 '{item['name']}' (索引 {idx})"
                else:
                    item_description = f"字典项目 (索引 {idx})"
            else:
                print(f"警告: {item_description} 没有 'boundary' 属性/键，面积计为0。")
                areas_list.append(0.0)
                continue

            if not isinstance(boundary_points, list):
                print(f"警告: {item_description} 的 'boundary' 不是列表类型，面积计为0。")
                areas_list.append(0.0)
                continue

            # 现在调用辅助函数计算面积，该函数内部会进行进一步的顶点验证
            area = DataUtils._calculate_polygon_area_shoelace(boundary_points)
            if area == 0.0 and (not boundary_points or len(boundary_points) < 3):
                # 只有当面积为0且确实是因为顶点不足或为空时才打印特定警告，
                # 其他情况（如顶点格式错误）的警告由 _calculate_polygon_area_shoelace 内部处理（如果取消注释）
                # 或通过其返回0.0来隐式处理。
                pass  # 可以选择在这里添加针对 boundary_points 的额外日志

            areas_list.append(area)

        return areas_list

    @staticmethod
    def find_connected_items(items: list, miss_items_names: list) -> list:
        """
        从一个包含多个item对象的列表中，根据未连接的item名称列表，找出已连接的item。

        Args:
            items (list): 一个item对象的列表。
                          每个item是一个对象，应具有 'name' (str) 属性。
                          例如，可以是自定义类的实例，如 class Item: def __init__(self, name, boundary): self.name = name; self.boundary = boundary
            miss_items_names (list): 一个包含未连接item名称的字符串列表。

        Returns:
            list: 一个包含已连接item对象的列表。
        """
        connected_items_list = []
        # 将 miss_items_names 转换为集合，以便进行更快的查找 (O(1) 平均时间复杂度)
        miss_items_set = set(miss_items_names)

        for item in items:
            # 检查 item 对象是否有 'name' 属性
            if hasattr(item, 'name'):
                item_name = item.name
                if item_name not in miss_items_set:
                    connected_items_list.append(item)
            else:
                # 记录没有 'name' 属性的项，这有助于调试
                print(f"警告: 项目 {item} (类型: {type(item)}) 没有 'name' 属性，将被忽略。")

        return connected_items_list

    @staticmethod
    def find_missing_items_in_paths(target_names, target_coords, source_names, source_coords, paths):
        """
        找出规划路径中缺失的起点和终点对应的 item 名称。

        Args:
            target_names (list): 终点 item 名称列表 (e.g., ['M14']).
            target_coords (list): 终点 x,y 坐标列表 (e.g., [[61, 53]]).
            source_names (list): 起点 item 名称列表 (e.g., ['M10', 'M6', 'M9']).
            source_coords (list): 起点 x,y 坐标列表 (e.g., [[47, 58], [30, 38], [39, 62]]).
            paths (list): 规划出的路径列表，每个路径是坐标元组的列表
                          (e.g., [[(20, 5), ..., (24, 5)], [(49, 26), ..., (21, 5)]]).

        Returns:
            list: 缺失的起点或终点对应的 item 名称列表。
        """
        missing_items = []

        # 确保输入坐标是元组形式，方便查找和比较
        # 并创建坐标到名称的映射字典
        target_coord_map = {tuple(coord): name for coord, name in zip(target_coords, target_names)}
        source_coord_map = {tuple(coord): name for coord, name in zip(source_coords, source_names)}

        # 提取 paths 中实际使用的起点和终点
        actual_start_points = set()
        actual_end_points = set()

        if paths:  # 检查 paths 是否为空列表
            for path in paths:
                if path and isinstance(path, collections.abc.Sequence) and len(path) > 0:
                    # 确保路径不为空且是序列
                    # 将路径的第一个点（起点）和最后一个点（终点）转换为元组并添加到集合中
                    # 假设path中的坐标已经是元组，如果不是，需要转换，例如 tuple(path[0])
                    try:
                        # 尝试直接添加，假设已经是元组
                        actual_start_points.add(path[0])
                        actual_end_points.add(path[-1])
                    except TypeError:
                        # 如果不是元组（比如是列表），则转换
                        if isinstance(path[0], list):
                            actual_start_points.add(tuple(path[0]))
                        if isinstance(path[-1], list):
                            actual_end_points.add(tuple(path[-1]))
                # else: # 可以选择处理空路径或无效路径的情况
                #    print(f"警告：跳过空路径或无效路径：{path}")

        # 检查预期的终点是否在实际的终点集合中
        for coord, name in target_coord_map.items():
            if coord not in actual_end_points:
                missing_items.append(name)

        # 检查预期的起点是否在实际的起点集合中
        for coord, name in source_coord_map.items():
            if coord not in actual_start_points:
                missing_items.append(name)

        return missing_items

    @staticmethod
    def sort_lists_based_on_first(list_to_sort_by, list_to_follow):

        # 1. 检查前提条件：确保两个列表长度相同
        assert len(list_to_sort_by) == len(list_to_follow), \
            "输入列表 list_to_sort_by 和 list_to_follow 的长度必须相同！"

        # 处理空列表的边界情况
        if not list_to_sort_by:
            return [], []

        # 2. 使用 zip 将两个列表的对应元素打包成元组列表
        #    例如：[(563.0, {'M3':...}), (1220.0, {'M3':...}), (820.66, {'M2':...})]
        combined_list = list(zip(list_to_sort_by, list_to_follow))

        # 3. 对打包后的列表进行排序
        #    sorted() 函数默认会根据元组的第一个元素（即 list_to_sort_by 中的值）进行排序
        #    可以使用 key 参数明确指定，但在此场景下非必需
        #    key=lambda item: item[0]
        sorted_combined_list = sorted(combined_list, key=lambda item: item[0])

        # 4. 解包（unzip）排序后的元组列表，得到两个新的已排序列表
        #    zip(*...) 是 zip 的逆操作
        #    map(list, ...) 将解包后的元组结果转换为列表
        if not sorted_combined_list:  # 再次检查以防万一（虽然前面已检查）
            return [], []
        sorted_list1, sorted_list2 = map(list, zip(*sorted_combined_list))

        return sorted_list1, sorted_list2

    @staticmethod
    def calculate_total_pairwise_manhattan_distance(points_list):
        # 计算列表中所有点两两之间的曼哈顿距离之和。
        n = len(points_list)
        # 如果点的数量少于2，无法形成点对，距离和为0
        if n < 2:
            return 0.0

        total_distance = 0.0

        # 使用嵌套循环遍历所有唯一的点对 (i, j)，确保 i < j
        # 这样可以避免重复计算（如 P1到P2 和 P2到P1）以及点与自身的距离
        for i in range(n):
            for j in range(i + 1, n):
                p1 = points_list[i]  # 第 i 个点
                p2 = points_list[j]  # 第 j 个点

                # 计算 p1 和 p2 之间的曼哈顿距离
                # distance = |x1 - x2| + |y1 - y2|
                distance = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

                # 将当前距离累加到总距离中
                total_distance += distance

        return total_distance


    @staticmethod
    def calculate_polygon_centroid(coordinates):
        if not coordinates:
            return None

        sum_x = 0.0
        sum_y = 0.0
        n = len(coordinates)  # 顶点数量

        for x, y in coordinates:
            sum_x += x
            sum_y += y

        centroid_x = sum_x / n
        centroid_y = sum_y / n

        return (centroid_x, centroid_y)


    @staticmethod
    def parse_txt_file(file_path):
        """
        解析包含 RotateItem 信息的旧格式文本文件。
        (保留函数，即使当前未调用)
        """
        area = None
        links = None
        name_list = None
        rotate_items = []
        current_ports = []

        try:
            with open(file_path, 'r', encoding='utf-8') as file: # 指定编码
                for line in file:
                    line = line.strip()
                    if not line: continue

                    if line.startswith("area:"):
                        try:
                            area = ast.literal_eval(line.split("area:", 1)[1].strip())
                        except Exception as e:
                            print(f"解析 area 出错: {e} in line: {line}")
                    elif line.startswith("links:"):
                         try:
                            links = ast.literal_eval(line.split("links:", 1)[1].strip())
                         except Exception as e:
                            print(f"解析 links 出错: {e} in line: {line}")
                    elif line.startswith("name_list:"):
                         try:
                            name_list = ast.literal_eval(line.split("name_list:", 1)[1].strip())
                         except Exception as e:
                            print(f"解析 name_list 出错: {e} in line: {line}")
                    elif line.startswith("rotate_item:"):
                        current_ports = [] # 开始新的 item，重置端口列表
                        item_data_str = line.split("rotate_item:", 1)[1].strip()
                        try:
                            match = re.search(r"(\{.*?\})(?:,\s*ports=.*)?$", item_data_str)
                            if match:
                                item_dict_str = match.group(1)
                                item_dict = ast.literal_eval(item_dict_str)
                                rotate_item = RotateItem(
                                    name=item_dict.get("name"),
                                    w=item_dict.get("w"),
                                    h=item_dict.get("h"),
                                    boundary=item_dict.get("boundary"),
                                    orient=item_dict.get("orient"),
                                    ports=current_ports
                                )
                                rotate_items.append(rotate_item)
                            else:
                                print(f"无法在 rotate_item 行中找到有效的字典格式: {line}")
                        except Exception as e:
                            print(f"解析 rotate_item 出错: {e} in line: {line}")

                    elif line.startswith("port:"):
                        port_data_str = line.split("port:", 1)[1].strip()
                        try:
                            port = ast.literal_eval(port_data_str)
                            if rotate_items:
                                rotate_items[-1].ports.append(port)
                            else:
                                print(f"警告: 找到 port 数据但没有关联的 rotate_item: {line}")
                        except Exception as e:
                            print(f"解析 port 出错: {e} in line: {line}")
        except FileNotFoundError:
            print(f"错误: 文件未找到 {file_path}")
            return None
        except Exception as e:
            print(f"读取文件时发生错误 {file_path}: {e}")
            return None

        return {"area": area, "links": links, "name_list": name_list, "rotate_items": rotate_items}

    @staticmethod
    def parse_txt_file_ForRAW(file_path):
        """
        解析包含 Item 信息的原始格式文本文件。
        (保留函数，即使当前未调用)
        """
        area = None
        links = None
        items = []
        current_item_details = {}

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if not line: continue

                    if line.startswith("links:"):
                        try: links = ast.literal_eval(line.split("links:", 1)[1].strip())
                        except Exception as e: print(f"解析 links 出错: {e} in line: {line}")
                    elif line.startswith("Area:"):
                        try:
                            area_data = line.split("Area:", 1)[1].strip().replace(')(', '), (')
                            area = ast.literal_eval(area_data)
                        except Exception as e: print(f"解析 Area 出错: {e} in line: {line}")
                    elif line.startswith("Rule:"): pass
                    elif line.startswith("Module:"):
                        if current_item_details.get('name'):
                            try:
                                item = Item(name=current_item_details['name'], boundary=current_item_details.get('boundary', []), ports=current_item_details.get('ports', []), boundary_layer=current_item_details.get('boundary_layer', None), port_layers=None)
                                items.append(item)
                            except Exception as e: print(f"创建 Item 对象时出错: {e} for {current_item_details.get('name')}")
                        current_item_details = {'name': line.split("Module:", 1)[1].strip(), 'ports': []}
                    elif line.startswith("Boundary:"):
                        if 'name' in current_item_details:
                            try:
                                parts = line.split("Boundary:", 1)[1].split(';', 1)
                                boundary_str = parts[0].strip()
                                coords = re.findall(r'\(\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\)', boundary_str)
                                current_item_details['boundary'] = [[float(x), float(y)] for x, y in coords]
                                if len(parts) > 1: current_item_details['boundary_layer'] = parts[1].strip()
                            except Exception as e: print(f"解析 Boundary 出错: {e} in line: {line}")
                        else: print(f"警告: 找到 Boundary 但没有当前模块: {line}")
                    elif line.startswith("Port:"):
                         if 'name' in current_item_details:
                            try:
                                parts = line.split("Port:", 1)[1].split(';', 1)
                                port_str = parts[0].strip()
                                coords = re.findall(r'\(\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\)', port_str)
                                port_info = {'coordinates': [(float(x), float(y)) for x, y in coords]}
                                if len(parts) > 1: port_info['type'] = parts[1].strip()
                                else: port_info['type'] = None
                                current_item_details['ports'].append(port_info)
                            except Exception as e: print(f"解析 Port 出错: {e} in line: {line}")
                         else: print(f"警告: 找到 Port 但没有当前模块: {line}")

            if current_item_details.get('name'):
                try:
                    item = Item(name=current_item_details['name'], boundary=current_item_details.get('boundary', []), ports=current_item_details.get('ports', []), boundary_layer=current_item_details.get('boundary_layer', None), port_layers=None)
                    items.append(item)
                except Exception as e: print(f"创建最后一个 Item 对象时出错: {e} for {current_item_details.get('name')}")

        except FileNotFoundError:
            print(f"错误: 文件未找到 {file_path}")
            return None
        except Exception as e:
            print(f"读取文件时发生错误 {file_path}: {e}")
            return None

        return {"area": area, "links": links, "items": items}

    @staticmethod
    def plot_rotate_items(rotate_items, area, flag=0):
        mpl.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams['axes.unicode_minus'] = False
        """绘制 RotateItem 对象 (旧格式) (保留函数)"""
        if not rotate_items: print("没有 RotateItem 可供绘制。"); return
        fig, ax = plt.subplots(figsize=(10, 8))
        if area:
             area_patch = patches.Polygon(area, closed=True, edgecolor='gray', facecolor='none', linestyle='--', lw=1)
             ax.add_patch(area_patch)
             min_x, max_x = min(p[0] for p in area) - 20, max(p[0] for p in area) + 20
             min_y, max_y = min(p[1] for p in area) - 20, max(p[1] for p in area) + 20
        else: # 估算范围
            all_coords = [p for item in rotate_items for p in item.boundary]
            if not all_coords: return
            min_x, max_x = min(p[0] for p in all_coords) - 20, max(p[0] for p in all_coords) + 20
            min_y, max_y = min(p[1] for p in all_coords) - 20, max(p[1] for p in all_coords) + 20

        for rotate_item in rotate_items:
            boundary = rotate_item.boundary
            if not boundary: continue
            try:
                boundary_polygon = patches.Polygon(boundary, closed=True, edgecolor='blue', facecolor='lightblue', alpha=0.6, lw=1)
                ax.add_patch(boundary_polygon)
                center_x = sum(p[0] for p in boundary) / len(boundary)
                center_y = sum(p[1] for p in boundary) / len(boundary)
                ax.text(center_x, center_y, rotate_item.name, ha='center', va='center', fontsize=8)
            except Exception as e: print(f"绘制模块 {rotate_item.name} 边界时出错: {e}")
            ports_to_draw = [p['coordinates'] for p in rotate_item.ports if p.get('coordinates')] if flag == 1 else rotate_item.ports
            for port_coords in ports_to_draw:
                 if len(port_coords) >= 3:
                    try:
                        port_polygon = patches.Polygon(port_coords, closed=True, edgecolor='orange', facecolor='yellow', alpha=0.7, lw=1)
                        ax.add_patch(port_polygon)
                    except Exception as e: print(f"绘制模块 {rotate_item.name} 的端口 {port_coords} 时出错: {e}")

        ax.set_aspect('equal', adjustable='box'); ax.set_xlim(min_x, max_x); ax.set_ylim(min_y, max_y)
        ax.set_title("模块布局与端口"); ax.set_xlabel("X 坐标"); ax.set_ylabel("Y 坐标")
        plt.grid(True, linestyle='--', alpha=0.5); plt.show()

    @staticmethod
    def plot_grid(grid):
        mpl.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams['axes.unicode_minus'] = False

        """绘制网格占用情况 (1=障碍, 2=端口) (保留函数)"""
        if grid is None or grid.size == 0: print("无法绘制空网格。"); return
        num_rows, num_cols = grid.shape
        plt.figure(figsize=(10, 8 * num_rows / max(1, num_cols)))
        cmap = plt.cm.colors.ListedColormap(['white', 'lightblue', 'lightgreen'])
        bounds = [-0.5, 0.5, 1.5, 2.5]; norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        plt.imshow(grid, cmap=cmap, norm=norm, origin='upper', extent=(-0.5, num_cols-0.5, -0.5, num_rows-0.5))
        ax = plt.gca(); ax.set_xticks(np.arange(-0.5, num_cols, 1), minor=True); ax.set_yticks(np.arange(-0.5, num_rows, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5); ax.tick_params(which='minor', size=0)
        ax.set_xticks(np.arange(0, num_cols, max(1, num_cols // 10))); ax.set_yticks(np.arange(0, num_rows, max(1, num_rows // 10)))
        plt.title("网格占用 (蓝色:障碍, 绿色:端口)"); plt.xlabel("列"); plt.ylabel("行")
        plt.colorbar(ticks=[0, 1, 2], format=plt.FuncFormatter(lambda val, loc: ['空闲', '障碍', '端口'][int(val)])); plt.show()

    @staticmethod
    def plot_grid_with_path(grid, all_paths, source_list_grid=None, target_list_grid=None, params=None):
        mpl.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams['axes.unicode_minus'] = False
        """
        绘制带有路径的网格图。
        (代码与上一版本相同，添加 params 用于获取 penalty)
        """
        if grid is None or grid.size == 0: print("无法绘制空网格。"); return

        penalty_factor = params.get('penalty_factor', 100000) if params else 100000 # 从参数获取惩罚因子

        valid, total_length, count = DataUtils.evaluate_paths(all_paths, source_list_grid or [], target_list_grid or [])
        score_str = f"Score: {total_length:.2f}"
        if not valid:
            if count > 0:
                 score_for_calc = total_length + count * penalty_factor
                 score_str = f"Score: {score_for_calc:.2f} (无效, 缺失 {count} 点, 基础长度 {total_length:.2f})"
            else:
                 score_for_calc = float('inf')
                 score_str = f"Score: 无穷大 (无效, 基础长度 {total_length:.2f})"
        else:
            score_for_calc = total_length

        num_rows, num_cols = grid.shape
        fig, ax = plt.subplots(figsize=(12, 10 * num_rows / max(1, num_cols)))

        cmap_bg = plt.cm.colors.ListedColormap(['#FFFFFF', '#A0A0A0', '#FFDB58'])
        bounds_bg = [-0.5, 0.5, 1.5, 2.5]; norm_bg = plt.cm.colors.BoundaryNorm(bounds_bg, cmap_bg.N)
        ax.imshow(grid, cmap=cmap_bg, norm=norm_bg, origin='upper', extent=(-0.5, num_cols-0.5, num_rows-0.5, -0.5), alpha=0.4)

        path_colors = plt.cm.viridis(np.linspace(0, 1, max(1, len(all_paths)))) # 至少一种颜色
        path_line_styles = ['-', '--', ':', '-.']

        for group_idx, path_group in enumerate(all_paths):
            color = path_colors[group_idx % len(path_colors)]
            for path_idx, path in enumerate(path_group):
                if not path: continue
                style = path_line_styles[path_idx % len(path_line_styles)]
                path_rows = [p[0] for p in path]; path_cols = [p[1] for p in path]
                ax.plot(path_cols, path_rows, color=color, linestyle=style, linewidth=1.5, marker='o', markersize=2, alpha=0.8, label=f'Link {group_idx+1} Path {path_idx+1}')
                if len(path) > 0:
                   ax.plot(path_cols[0], path_rows[0], 'X', color=color, markersize=6, alpha=0.9)
                   ax.plot(path_cols[-1], path_rows[-1], 's', color=color, markersize=5, alpha=0.9)

        if source_list_grid:
            all_sources = set(tuple(p) for group in source_list_grid for p in group)
            source_rows = [p[0] for p in all_sources]; source_cols = [p[1] for p in all_sources]
            ax.scatter(source_cols, source_rows, marker='^', color='red', s=50, label='Source Ports', alpha=0.7, zorder=5)
        if target_list_grid:
            all_targets = set(tuple(p) for group in target_list_grid for p in group)
            target_rows = [p[0] for p in all_targets]; target_cols = [p[1] for p in all_targets]
            ax.scatter(target_cols, target_rows, marker='v', color='blue', s=50, label='Target Ports', alpha=0.7, zorder=5)

        ax.set_xlim(-0.5, num_cols - 0.5); ax.set_ylim(num_rows - 0.5, -0.5)
        ax.set_xticks(np.arange(-0.5, num_cols, 1), minor=True); ax.set_yticks(np.arange(-0.5, num_rows, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.5); ax.tick_params(which='minor', size=0)
        ax.set_xticks(np.arange(0, num_cols, max(1, num_cols // 10))); ax.set_yticks(np.arange(0, num_rows, max(1, num_rows // 10)))
        # ax.set_title(f"网格布线结果 ({score_str})"); ax.set_xlabel("列"); ax.set_ylabel("行")
        # ax.set_title(f""); ax.set_xlabel("x"); ax.set_ylabel("y")
        # plt.tight_layout()

        # 隐藏坐标轴的边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # 隐藏刻度标签（数字）
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # 隐藏刻度线
        ax.tick_params(axis='both', which='both', length=0)

        # 获取图的边界
        x_min, x_max = ax.get_xlim()
        y_max, y_min = ax.get_ylim()  # 注意y轴是反向的，所以y_max对应图的上边界，y_min对应下边界

        # 创建一个与图的边界重合的矩形作为外边框
        from matplotlib.patches import Rectangle
        rect_width = x_max - x_min
        rect_height = y_max - y_min  # 由于y轴是反向的，这里计算高度

        # 边框的起始点是 (x_min, y_min)
        outer_border = Rectangle(
            (x_min, y_min),  # 左下角坐标
            rect_width,  # 宽度
            rect_height,  # 高度
            linewidth=2,  # 线宽
            edgecolor='black',  # 边框颜色
            facecolor='none',  # 内部填充颜色（无填充）
            zorder=10  # 确保边框在最上层显示
        )
        ax.add_patch(outer_border)

        plt.show()


    @staticmethod
    def reorder_multiple_lists(lists, method="random", key=None, reverse=False):
        """
        对多个长度相同的列表按照同一规则重新排序。
        (代码与上一版本相同)
        """
        if not lists: return []
        try:
            first_len = len(lists[0])
            if not all(len(lst) == first_len for lst in lists[1:]):
                 raise ValueError("所有列表长度必须相同！")
        except IndexError: # 处理空列表的情况
             return [[] for _ in lists]


        num_items = first_len
        if num_items == 0: return [[] for _ in lists] # 如果列表为空，直接返回空列表

        indices = list(range(num_items))

        if method == "random":
            random.shuffle(indices)
        elif method == "custom" and key:
            try: indices = sorted(indices, key=key, reverse=reverse)
            except Exception as e: raise ValueError(f"自定义排序键函数出错: {e}")
        elif method == "custom": raise ValueError("自定义排序需要提供 key 函数。")
        elif method == "swap":
            if num_items >= 2:
                idx1, idx2 = random.sample(range(num_items), 2)
                indices[idx1], indices[idx2] = indices[idx2], indices[idx1]
        else: raise ValueError(f"不支持的排序方法: {method}")

        reordered_lists = [[lst[i] for i in indices] for lst in lists]
        return reordered_lists

    def check_missing_coordinates(all_paths, source_list_grid, target_list_grid):
        # 扁平化 source_list_grid 和 target_list_grid
        source_coords = [tuple(item) for sublist in source_list_grid for item in sublist]
        target_coords = [tuple(item) for sublist in target_list_grid for item in sublist]

        # 用来存放漏掉的坐标和缺失标志
        missing_coords = {
            'source': {
                'coords': [],
                'isMissing': False,
                'count': 0
            },
            'target': {
                'coords': [],
                'isMissing': False,
                'count': 0
            }
        }

        # 扁平化所有路径的坐标
        all_path_coords = [coord_tuple for path in all_paths for sub_path in path for coord_tuple in sub_path]

        # 检查 source_list_grid 中的坐标是否在 all_path 中
        for coord in source_coords:
            if coord not in all_path_coords:
                missing_coords['source']['coords'].append(coord)
                missing_coords['source']['isMissing'] = True
                missing_coords['source']['count'] += 1

        # 检查 target_list_grid 中的坐标是否在 all_path 中
        for coord in target_coords:
            if coord not in all_path_coords:
                missing_coords['target']['coords'].append(coord)
                missing_coords['target']['isMissing'] = True
                missing_coords['target']['count'] += 1

        return missing_coords

    def evaluate_paths(all_paths, source_list_grid, target_list_grid):
        """
        评估路径结果：
        1. 如果任何路径为空，则舍弃。
        2. 如果所有路径非空，计算总路径长度（曼哈顿距离）。

        参数:
            all_paths (list): 所有路径的列表，其中每个路径是点坐标的列表。

        返回:
            valid (bool): 是否有效路径结果（无空路径）。
            total_length (int): 所有路径的总长度（仅当有效时计算）。
        """
        # 检查是否有空路径
        # for paths in all_paths:
        #     if not paths:  # 判断空列表
        #         return False, 0 , 0

        # 检查子路径中是否有漏掉的端口点
        missing_coords = DataUtils.check_missing_coordinates(all_paths, source_list_grid, target_list_grid)
        if missing_coords['source']['isMissing'] or missing_coords['target']['isMissing']:
            c = missing_coords['source']['count'] + missing_coords['target']['count']
            return False, 0, c

        total_length = 0
        for paths in all_paths:
            for path in paths:
                if len(path) < 2:  # 如果路径长度小于2，跳过
                    continue
                for i in range(len(path) - 1):
                    point1 = path[i]
                    point2 = path[i + 1]
                    # 检查点是否为坐标对
                    if not (isinstance(point1, tuple) and isinstance(point2, tuple)):
                        raise ValueError("Each path must consist of coordinate tuples (x, y).")
                    total_length += abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])  # 曼哈顿距离

        return True, total_length, 0
    @staticmethod
    def getScore(items, area, path):
        print("items:",items[0])
        print("area:", area)
        print("path:", path)
        print("end")
        pass



# --- 重构后的函数 ---

def load_data(links_path, items_path):
    """
    加载 links 和 items/area 数据。

    Args:
        links_path (str): links 文件路径。
        items_path (str): items 文件路径。

    Returns:
        tuple: (links, area, items) 或 (None, None, None) 如果加载失败。
    """
    print(f"加载连接文件: {links_path}")
    links = ReadFileUtils.parse_links_file(links_path)
    print(f"加载布局文件: {items_path}")
    area, items = ReadFileUtils.parse_items_file(items_path)
    print("area", area)
    print("items", items)
    # item坐标需要根据area左下角点进行修正
    x = area[0][0]
    y = area[0][1]
    for item in items:
        for coor in item.boundary:
            coor[0] -= area[0][0]
            coor[1] -= area[0][1]
        for port in item.ports:
            for i, coor in enumerate(port['coordinates']):
                port['coordinates'][i] = (coor[0] - area[0][0], coor[1] - area[0][1])

    area = list(area)
    for i, coor in enumerate(area):
        area[i] = (coor[0] - x, coor[1] - y)

    if links is None or area is None or items is None:
        print("错误：加载数据失败，请检查文件路径和格式。")
        return None, None, None
    print(f"--- 数据加载完成 (Links: {len(links)}, Items: {len(items)}) ---")
    return links, area, items

# def setup_parameters():
#     """
#     设置并返回所有配置参数。
#
#     Returns:
#         dict: 包含所有参数的字典。
#     """
#     grid_size = 20
#     params = {
#         'grid_size': grid_size,
#         'port_boundary_index': 0,
#         'affected_radius': 2 * grid_size,
#         'affected_value_for_item': 2000.0,
#         'affected_value_for_path': 100.0,
#         'used_value': 10000.0,
#         'history_length': 15,
#         'max_iterations': 50,
#         'penalty_factor': 100000.0
#     }
#     print("--- 参数设置 ---")
#     for key, value in params.items():
#         print(f"  {key}: {value}")
#     return params


# --- 辅助函数：用于从文件名中提取配置标识符 ---
def extract_config_identifier(links_path):
    """
    从 links 文件名中提取数字标识符。
    例如：'connect_40.txt' -> 40
    """
    try:
        filename = os.path.basename(links_path) # 例如：connect_40.txt
        # 使用正则表达式查找 'connect_'之后、'.txt'之前的数字
        match = re.search(r'connect_(\d+)', filename)
        if match:
            return int(match.group(1))
        else:
            print(f"警告：无法从 '{filename}' 中提取标识符。将使用默认参数。")
            return None # 或者返回一个默认标识符，比如 0
    except Exception as e:
        print(f"从 '{links_path}' 提取标识符时出错：{e}。将使用默认参数。")
        return None

# --- 修改后的 setup_parameters ---
def setup_parameters(config_identifier=None): # 添加了 config_identifier 参数
    """
    设置并返回所有配置参数。

    Args:
        config_identifier (int, optional): 从文件名中提取的标识符 (例如, connect_40 中的 40)。
                                           默认为 None, 此时将使用默认参数。

    Returns:
        dict: 包含所有参数的字典。
    """
    # 默认参数
    grid_size = 5
    params = {
        'grid_size': grid_size,
        'port_boundary_index': 0,
        'affected_radius': 2 * grid_size, # 如果 grid_size 改变，这个值会被更新
        'affected_value_for_item': 2000.0,
        'affected_value_for_path': 200.0,
        'used_value': 10000.0,
        'history_length': 15,
        'max_iterations': 50,
        'penalty_factor': 100000.0
    }

    # print(f"--- 参数设置 (标识符: {config_identifier}) ---")

    # 根据 config_identifier 调整参数
    # 快速的
    if config_identifier == 3:
        print("应用配置 'connect_5' 的参数")
        params['grid_size'] = 15
        params['max_iterations'] = 50
        params['history_length'] = 25
    elif config_identifier == 6:
        print("应用配置 'connect_10' 的参数")
        # params['grid_size'] = 10
        params['grid_size'] = 20
        params['max_iterations'] = 50
        params['history_length'] = 25
    elif config_identifier == 8:
        print("应用配置 'connect_16' 的参数")
        # params['grid_size'] = 10
        params['grid_size'] = 20
        params['max_iterations'] = 50
        params['history_length'] = 25
    elif config_identifier == 9:
        print("应用配置 'connect_20' 的参数")
        # params['grid_size'] = 15
        params['grid_size'] = 20
        params['max_iterations'] = 50
        params['history_length'] = 15
    elif config_identifier == 12:
        print("应用配置 'connect_25' 的参数")
        params['grid_size'] = 20
        params['max_iterations'] = 50
        params['history_length'] = 25
    elif config_identifier == 15:
        print("应用配置 'connect_30' 的参数")
        params['grid_size'] = 20
        params['max_iterations'] = 50
        params['history_length'] = 25
    # elif config_identifier == 18:
    #     print("应用配置 'connect_35' 的参数")
    #     params['grid_size'] = 15
    #     params['max_iterations'] = 50
    #     params['history_length'] = 25
    elif config_identifier == 26:
        print("应用配置 'connect_40' 的参数")
        params['grid_size'] = 25
        params['max_iterations'] = 50
        params['history_length'] = 25
    elif config_identifier == 24:
        print("应用配置 'connect_45' 的参数")
        params['grid_size'] = 25
        params['max_iterations'] = 50
        params['history_length'] = 25
    # elif config_identifier == 28:
    #     print("应用配置 'connect_50' 的参数")
    #     params['grid_size'] = 20
    #     params['max_iterations'] = 50
    #     params['history_length'] = 25
    else:
        if config_identifier is not None: # 仅当传入了ID但未匹配时发出警告
            print(f"警告：未找到标识符 '{config_identifier}' 的特定参数。将使用默认/基础参数。")
        else:
            print("使用默认/基础参数。")

    # # 慢的
    # if config_identifier == 3:
    #     print("应用配置 'connect_5' 的参数")
    #     params['grid_size'] = 5
    #     params['max_iterations'] = 50
    #     params['history_length'] = 25
    # elif config_identifier == 6:
    #     print("应用配置 'connect_10' 的参数")
    #     params['grid_size'] = 10
    #     params['max_iterations'] = 50
    #     params['history_length'] = 25
    # elif config_identifier == 8:
    #     print("应用配置 'connect_16' 的参数")
    #     params['grid_size'] = 10
    #     params['max_iterations'] = 50
    #     params['history_length'] = 25
    # elif config_identifier == 9:
    #     print("应用配置 'connect_20' 的参数")
    #     params['grid_size'] = 15
    #     params['max_iterations'] = 50
    #     params['history_length'] = 15
    # elif config_identifier == 12:
    #     print("应用配置 'connect_25' 的参数")
    #     params['grid_size'] = 20
    #     params['max_iterations'] = 50
    #     params['history_length'] = 25
    # elif config_identifier == 15:
    #     print("应用配置 'connect_30' 的参数")
    #     params['grid_size'] = 20
    #     params['max_iterations'] = 50
    #     params['history_length'] = 25
    # # elif config_identifier == 18:
    # #     print("应用配置 'connect_35' 的参数")
    # #     params['grid_size'] = 15
    # #     params['max_iterations'] = 50
    # #     params['history_length'] = 25
    # elif config_identifier == 26:
    #     print("应用配置 'connect_40' 的参数")
    #     params['grid_size'] = 25
    #     params['max_iterations'] = 50
    #     params['history_length'] = 25
    # elif config_identifier == 24:
    #     print("应用配置 'connect_45' 的参数")
    #     params['grid_size'] = 25
    #     params['max_iterations'] = 50
    #     params['history_length'] = 25
    # # elif config_identifier == 28:
    # #     print("应用配置 'connect_50' 的参数")
    # #     params['grid_size'] = 20
    # #     params['max_iterations'] = 50
    # #     params['history_length'] = 25
    # else:
    #     if config_identifier is not None:  # 仅当传入了ID但未匹配时发出警告
    #         print(f"警告：未找到标识符 '{config_identifier}' 的特定参数。将使用默认/基础参数。")
    #     else:
    #         print("使用默认/基础参数。")


    # 关键：如果 grid_size 被更改，则更新依赖于它的参数
    if params['grid_size'] != grid_size: # 如果 grid_size 从默认值更新了
        params['affected_radius'] = 2 * params['grid_size']
        # 可能还需要更新其他依赖于 grid_size 的参数

    # for key, value in params.items():
    #     print(f"  {key}: {value}")
    return params


def initialize_routing(rs, items, area, links, params):
    """
    初始化布线所需的数据结构。

    Args:
        rs (RoutingScore): RoutingScore 实例。
        items (list): Item 对象列表。
        area (list): 区域边界。
        links (dict): 连接信息列表。
        params (dict): 参数字典。

    Returns:
        tuple: (source_raw, target_raw, source_grid, target_grid, grid_list, weight_list)
               或 None 如果初始化失败。
    """
    # print("初始化源/目标点、网格和权重...")
    # try:
    # 时间
    start_time = datetime.now()

    links_2 = []  # 用于存放所有值都为 '2' 的元素
    links_13 = []  # 用于存放其余的元素

    # 遍历原始 links 列表
    for item in links:
        # 检查字典是否非空且所有值是否都为 '2' 即找出第二层的item
        if item and all(value == '2' for value in item.values()):
            # 如果是，添加到 links_2
            links_2.append(item)
        else:
            # 如果不是，添加到 links_13
            links_13.append(item)


    source_13_raw, target_13_raw, source_list_13_grid, target_list_13_grid, grid_13_list, weight_13_list, modulelist_13_Target, modulelist_13_Source, connected_13_ports = rs.getSourceTargetFor(
        items, area, links_13,
        params['grid_size'], params['affected_radius'],
        params['affected_value_for_item'], params['used_value'],
        params['port_boundary_index'], 2
    )
    source_2_raw, target_2_raw, source_list_2_grid, target_list_2_grid, grid_2_list, weight_2_list, modulelist_2_Target, modulelist_2_Source, connected_2_ports = rs.getSourceTargetFor(
        items, area, links_2,
        params['grid_size'], params['affected_radius'],
        params['affected_value_for_item'], params['used_value'],
        params['port_boundary_index'], 2
    )
    # 为第13层重新获取一次grid_13_list, weight_13_list 因为他们跟版图不在同一层
    grid_13_list = []
    weight_13_list = []
    for i in range(len(links_13)):
        grid, weight = rs.set_grid_weights_for_source_target_custom_corners(area, items, params['grid_size'], links_13,
                                                                              params['affected_radius'],
                                                                              params['affected_value_for_item'],
                                                                              params['used_value'], Flag=13)
        grid_13_list.append(grid)
        weight_13_list.append(weight)

    # 按照links_connected_ports对link进行排序
    link_dist_list = []
    for connected_ports in connected_13_ports:
        # 针对每个connected_port计算出其对应的连接距离
        gravity_ports_list = []
        for port in connected_ports:
            gravity = DataUtils.calculate_polygon_centroid(port['port_coordinates'])
            gravity_ports_list.append((gravity))

        dist = DataUtils.calculate_total_pairwise_manhattan_distance(gravity_ports_list)
        link_dist_list.append(dist)
    link_dist_list, links_13 = DataUtils.sort_lists_based_on_first(link_dist_list, links_13)

    link_dist_list = []
    for connected_ports in connected_2_ports:
        # 针对每个connected_port计算出其对应的连接距离
        gravity_ports_list = []
        for port in connected_ports:
            gravity = DataUtils.calculate_polygon_centroid(port['port_coordinates'])
            gravity_ports_list.append((gravity))

        dist = DataUtils.calculate_total_pairwise_manhattan_distance(gravity_ports_list)
        link_dist_list.append(dist)
    link_dist_list, links_2 = DataUtils.sort_lists_based_on_first(link_dist_list, links_2)

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    # print(f"getSourceTarget: {elapsed_time.total_seconds():.2f} 秒")

    # 检查初始化结果是否有效
    if  len(source_list_13_grid) != len(links_13) or len(target_list_13_grid) != len(links_13) or \
        len(grid_13_list) != len(links_13) or len(weight_13_list) != len(links_13):
            print("错误：初始化未能为所有 13层 links 生成有效的网格或源/目标列表。")
            return None
    if  len(source_list_2_grid) != len(links_2) or len(target_list_2_grid) != len(links_2) or \
        len(grid_2_list) != len(links_2) or len(weight_2_list) != len(links_2):
            print("source_list_2_grid 长度", len(source_list_2_grid))
            print("target_list_2_grid 长度", len(target_list_2_grid))
            print("grid_2_list 长度", len(grid_2_list))
            print("weight_2_list 长度", len(weight_2_list))
            print("links_2 长度", len(links_2))

            print("错误：初始化未能为所有 2层 links 生成有效的网格或源/目标列表。")
            return None
    # print("--- 初始化完成 ---")
    return (
        source_13_raw, target_13_raw, source_list_13_grid, target_list_13_grid,
        grid_13_list, weight_13_list, modulelist_13_Target, modulelist_13_Source,
        links_13,  # 这是排序后的 links_13

        source_2_raw, target_2_raw, source_list_2_grid, target_list_2_grid,
        grid_2_list, weight_2_list, modulelist_2_Target, modulelist_2_Source,
        links_2  # 这是排序后的 links_2
    )

def calculate_score(solution, source_grid, target_grid, params):
    """
    计算给定解决方案的分数。

    Args:
        solution (list): 布线路径 [[path1, path2], ...]。
        source_grid (list): 原始请求的源点网格坐标列表。
        target_grid (list): 原始请求的目标点网格坐标列表。
        params (dict): 参数字典 (需要 'penalty_factor')。

    Returns:
        float: 计算得到的分数 (无效解为极大值或无穷大)。
    """
    valid, total_length, missing_count = DataUtils.evaluate_paths(solution, source_grid, target_grid)
    if valid:
        return total_length
    else:
        penalty_factor = params.get('penalty_factor', 100000.0)
        if missing_count > 0:
             # print(f"  评估: 无效 (缺失 {missing_count} 点), 长度 {total_length:.2f}, 惩罚分数 {total_length + missing_count * penalty_factor:.2f}")
             return total_length + missing_count * penalty_factor
        else:
             # print(f"  评估: 无效 (路径格式等问题), 长度 {total_length:.2f}, 分数 无穷大")
             return float('inf') # 无效但无缺失（例如路径格式错误）


def calculate_chip_score(
        items,  # list of all item objects (used for count)
        items_area_list,  # list of floats, areas for all items
        connected_items,  # list of successfully connected item objects (used for count)
        connected_items_area_list,  # list of floats, areas for successfully connected items
        success_area_size,  # float, area enclosed by successful modules and routing
        success_path_length,  # float, total wirelength of successful modules
        SuccessRate_WiringConnection,
        old_links,
        # float, e.g., 0.95 for 95% connection success
        k_wl_param=0.01, # 线长惩罚调节参数
):
    """
    计算芯片版图放置布线的评分。

    评分规则：
    1. 每个case分数包含三部分 (score_part1, score_part1_2, score_part1_3)
    1.1 score_part1: 基础放置布线分数
        - 如果所有模块都成功（即 len(connected_items) == len(items)）：
            - 分数根据总模块数量 (total_module_count) 决定：
                - total_module_count < 25: 50分
                - 25 <= total_module_count <= 40: 70分
                - total_module_count > 40: 90分
        - 如果未能全部成功：
            - 分数 = Case布线成功分数 * 模块利用率 * 模块面积利用率
            - Case布线成功分数：同上，根据总模块数量决定。
            - 模块利用率 = 成功模块数 / 总模块数量
            - 模块面积利用率 = 成功模块面积 / 总模块面积

    条件：以下条件满足才能获得 score_part1_2 和 score_part1_3，否则这两部分为0。
        - 总模块数量 < 25: 最多两个模块失败 (即 成功模块数 >= 总模块数量 - 2)
        - 25 <= 总模块数量 <= 50: 90% 模块成功率 (即 成功模块数 / 总模块数量 >= 0.90)
        - 总模块数量 > 50: 85% 模块成功率 (即 成功模块数 / 总模块数量 >= 0.85)

    1.2 score_part1_2: 成功模块和布线围成的面积分数 (若条件满足)
        - 分数 = 10 * (成功模块和布线围成的面积) / (模块面积利用率)
        - 模块面积利用率 = 成功模块面积 / 总模块面积

    1.3 score_part1_3: 成功模块的总线长分数 (若条件满足)
        - 分数 = 10 * (成功模块的总布线长度) / (布线连接成功率)

    Args:
        items (list): 所有模块对象的列表。
        items_area_list (list): 对应 `items` 中每个模块的面积列表 (float)。
        connected_items (list): 成功布局布线的模块对象的列表。
        connected_items_area_list (list): 对应 `connected_items` 中每个模块的面积列表 (float)。
        success_area_size (float): 成功模块和布线围成的总面积。
        success_path_length (float): 成功模块的总布线长度。
        SuccessRate_WiringConnection (float): 布线连接成功率 (例如 0.9 表示 90%)。

    Returns:
        float: 计算得到的总评分。
    """

    # --- 计算基础统计数据 ---
    #传进来的items并非完整的items，要拿新的
    total_module_count = DataUtils().get_max_item_number_from_links_regex(old_links)
    successful_module_count = len(connected_items)

    # 处理没有模块的特殊情况，评分为0
    if total_module_count == 0:
        return 0.0

    # 计算总面积和成功模块面积
    # 即使列表为空，sum([]) 也会返回0，是安全的
    total_module_area = sum(items_area_list)
    successful_module_area = sum(connected_items_area_list)

    # --- 计算通用的利用率 ---
    # 模块数量利用率 (total_module_count 在此已保证 > 0)
    module_utilization_rate = successful_module_count / total_module_count

    # 模块面积利用率
    module_area_utilization_rate = 0.0
    if total_module_area > 0:
        module_area_utilization_rate = successful_module_area / total_module_area

    # --- 第1部分: 基础放置布线分数 (score_part1) ---
    score_part1 = 0.0

    # 根据总模块数量确定基础布线分数
    base_routing_score = 0
    if total_module_count < 25:
        base_routing_score = 50
    elif 25 <= total_module_count <= 40:
        base_routing_score = 70
    else:  # total_module_count > 40
        base_routing_score = 90

    all_modules_successful = (successful_module_count == total_module_count)

    if all_modules_successful:
        # 如果所有模块都成功，直接获得基础布线分数
        score_part1 = float(base_routing_score)
    else:
        # 如果部分成功，则按比例计算
        score_part1 = base_routing_score * module_utilization_rate * module_area_utilization_rate

    # --- 判断是否满足获取1.2和1.3部分分数的条件 ---
    bonus_scores_eligible = False
    # total_module_count 在此已保证 > 0
    if total_module_count < 25:
        if successful_module_count >= total_module_count - 2:  # 最多两个失败
            bonus_scores_eligible = True
    elif 25 <= total_module_count <= 50:
        if module_utilization_rate >= 0.90:  # 90% 成功率
            bonus_scores_eligible = True
    else:  # total_module_count > 50
        if module_utilization_rate >= 0.85:  # 85% 成功率
            bonus_scores_eligible = True

    # --- 新的评分规则 1.2 (面向更高布局密度) ---
    score_part1_2 = 0.0
    if bonus_scores_eligible:
        if success_area_size > 0 and successful_module_area >= 0:  # 确保分母不为0，分子非负
            density = successful_module_area / success_area_size
            score_part1_2 = 10.0 * density
            # 限制最高分，防止 density > 1 (虽然理论上应该<=1，但浮点数可能有意外)
            score_part1_2 = min(score_part1_2, 10.0)
            # else: 保持 score_part1_2 = 0.0

    # --- 新的评分规则 1.3 (面向更高连接成功率和更短线长) ---
    score_part1_3 = 0.0
    if bonus_scores_eligible:
        avg_wl_per_module = 0.0
        if successful_module_count > 0 and success_path_length >= 0:  # 确保分子非负，分母大于0
            avg_wl_per_module = success_path_length / successful_module_count

        wirelength_penalty_factor = math.exp(-k_wl_param * avg_wl_per_module)

        score_part1_3 = 10.0 * SuccessRate_WiringConnection * wirelength_penalty_factor

    # --- 计算总分 ---
    total_score = score_part1 + score_part1_2 + score_part1_3
    return total_score

def run_lahc_optimization(rs, items, area, links, initial_data, params, old_links, start_time, max_time):
    """
    执行 LAHC 优化循环。
    """
    if initial_data is None:
        print("错误: 初始数据无效，无法开始 LAHC。")
        return [], float('inf')

    source_grid, target_grid, grid_list, weight_list, modulelist_Target, modulelist_Source = initial_data[2:] # 解包需要的部分

    # --- 生成并评估初始解 ---
    # print("生成初始布线解...")
    initial_input_data = [copy.deepcopy(source_grid), copy.deepcopy(target_grid),
                          copy.deepcopy(grid_list), copy.deepcopy(weight_list), copy.deepcopy(links)]

    # 为了计算评分，需要如下输出
    # 输出为总共的模块list、以及对应的面积list、布局成功的模块list、以及对应的面积list、成功模块和布线围成的面积、成功模块的总布线长度、布线连接成功率
    return_vals = rs.run_for_getOneSolution(
        area, items, copy.deepcopy(initial_input_data[0]), copy.deepcopy(initial_input_data[1]), copy.deepcopy(initial_input_data[2]),
        copy.deepcopy(initial_input_data[3]), copy.deepcopy(initial_input_data[4]), params, modulelist_Target, modulelist_Source, start_time, max_time
    )
    (items, items_area_list, connected_items, connected_items_area_list,
     success_area_size, success_path_length, SuccessRate_WiringConnection, current_solution, fail_item_list) = return_vals
    # count_linked_items_num = (len(items) - len(fail_item_list))
    # print("=================================成功items如下")
    # for item in connected_items:
    #     print(item.to_stringbyczg())
    # print("=================================成功数量==", count_linked_items_num)

    # print("评估初始解...")
    current_score = calculate_chip_score(items, items_area_list, connected_items, connected_items_area_list,
     success_area_size, success_path_length, SuccessRate_WiringConnection, old_links)
    # current_score = calculate_score(current_solution, source_grid, target_grid, params)
    # print(f"初始解分数: {current_score:.2f}")


    best_solution = copy.deepcopy(current_solution)
    best_score = current_score
    best_return_vals = copy.deepcopy(return_vals)

    # --- LAHC 循环 ---
    # print(f"\n--- 开始 LAHC 优化 (迭代次数: {params['max_iterations']}, 历史长度: {params['history_length']}) ---")
    history = [best_score] * params['history_length']
    history_index = 0
    accepted_input_data = initial_input_data # 从初始解的输入开始

    for iteration in range(params['max_iterations']):
        # 扰动上一次接受的输入数据
        perturb_input_data = copy.deepcopy(accepted_input_data)
        perturbed_lists = DataUtils.reorder_multiple_lists(perturb_input_data, method="swap")
        source_grid_iter, target_grid_iter, grid_list_iter, weight_list_iter, links_iter = perturbed_lists

        # 生成新解
        return_vals = rs.run_for_getOneSolution(
            area, items, copy.deepcopy(source_grid_iter), copy.deepcopy(target_grid_iter),
            copy.deepcopy(grid_list_iter), copy.deepcopy(weight_list_iter), copy.deepcopy(links_iter), params, modulelist_Target, modulelist_Source, start_time, max_time
        )
        (items, items_area_list, connected_items, connected_items_area_list,
         success_area_size, success_path_length, SuccessRate_WiringConnection, new_solution,
         fail_item_list) = return_vals
        count_linked_items_num = (len(items) - len(fail_item_list))
        # print("=================================成功items如下")
        # for item in connected_items:
        #     print(item.to_stringbyczg())
        #Todo: 注释
        #print("=================================成功数量==", count_linked_items_num)

        # 评估新解
        new_score = calculate_chip_score(items, items_area_list, connected_items, connected_items_area_list,
                                             success_area_size, success_path_length, SuccessRate_WiringConnection, old_links)
        # new_score = calculate_score(new_solution, source_grid, target_grid, params) # 始终用原始 source/target 评估

        # LAHC 接受准则
        historical_score_idx = (history_index - 1 + params['history_length']) % params['history_length']
        historical_score = history[historical_score_idx]

        accepted = False
        if new_score > current_score or new_score > historical_score:
            accepted = True
            current_solution = copy.deepcopy(new_solution)
            current_score = new_score
            current_return_vals = copy.deepcopy(return_vals)
            accepted_input_data = perturbed_lists # 保存接受解的输入数据

            # if current_score < best_score or count_linked_items_num > best_count_linked_items_num:
            if current_score > best_score:
                best_solution = copy.deepcopy(current_solution)
                best_score = current_score
                best_return_vals = copy.deepcopy(current_return_vals)
                # Todo: 注释
                # print(f"迭代 {iteration + 1}/{params['max_iterations']}: *新最优解* Score = {best_score:.2f} (对比历史: {historical_score:.2f})")
            else:
                # Todo: 注释
                #print(f"迭代 {iteration + 1}/{params['max_iterations']}: 接受新解   Score = {current_score:.2f} (对比历史: {historical_score:.2f})")
                pass
        else:
            # Todo: 注释
            #print(f"迭代 {iteration + 1}/{params['max_iterations']}: 拒绝新解   Score = {new_score:.2f} (当前: {current_score:.2f}, 历史: {historical_score:.2f})")
            pass # 减少打印信息


        # 更新历史记录 (使用当前接受的分数)
        history[history_index] = current_score
        history_index = (history_index + 1) % params['history_length']

    # print("\n--- LAHC 优化完成 ---")
    return best_solution, best_score, best_return_vals

def plot_final_result(rs, best_solution, initial_data, area, items, params):
    """
    绘制最终的最优布线结果。

    Args:
        rs (RoutingScore): RoutingScore 实例。
        best_solution (list): 最优布线路径。
        initial_data (tuple): 包含初始 source_grid, target_grid 的元组。
        area (tuple): 区域边界。
        items (list): Item 对象列表。
        params (dict): 参数字典。
    """
    if initial_data is None:
        print("错误: 初始数据无效，无法绘制结果。")
        return

    source_grid, target_grid = initial_data[2:4] # 解包需要的部分

    print("绘制最终最优解图...")
    try:
        # 创建用于绘图的基础网格 (只包含障碍)
        grid_raw_for_plot = rs.set_grid_weights(area, items, params['grid_size'])
        if grid_raw_for_plot is None or grid_raw_for_plot.size == 0:
             print("错误：无法生成用于绘图的基础网格。")
             return

        # 论文画图所需-不画端口-下面备注掉了
        # # 在基础网格上标记端口
        # rows, cols = grid_raw_for_plot.shape
        # all_port_coords = set(tuple(coord) for group in source_grid for coord in group)
        # all_port_coords.update(set(tuple(coord) for group in target_grid for coord in group))
        #
        # for r, c in all_port_coords:
        #      if 0 <= r < rows and 0 <= c < cols and grid_raw_for_plot[r,c] != 1: # 仅在非障碍处标记
        #          grid_raw_for_plot[r,c] = 2 # 标记为端口

        # DataUtils.plot_grid_with_path(grid_raw_for_plot, best_solution, source_grid, target_grid, params)
        DataUtils.plot_grid_with_path(grid_raw_for_plot, best_solution, [], [], params)
    except Exception as e:
        print(f"绘制最终结果时出错: {e}")



def transform_item_list(list_of_items):
    """
    转换 item_list 中每个 Item 对象的 ports 属性。
    ports 列表中的每个端口将从坐标列表转换为包含 'coordinates' 和 'type' 的字典。
    坐标点 [x,y] 会被转换为元组 (x,y)。
    这个函数会直接修改列表中的 Item 对象。
    """
    for item_object in list_of_items:
        if not hasattr(item_object, 'ports') or not hasattr(item_object, 'port_layers'):
            print(f"警告: Item '{item_object.name}' 缺少 'ports' 或 'port_layers' 属性。")
            continue

        original_ports_data = item_object.ports
        port_types = item_object.port_layers
        item_object.boundary = [list(coord_pair) for coord_pair in item_object.boundary]
        # 确保 original_ports_data 和 port_types 是列表，并且长度一致
        if not isinstance(original_ports_data, list) or not isinstance(port_types, list):
            print(f"警告: Item '{item_object.name}' 的 'ports' 或 'port_layers' 不是列表类型。")
            continue

        # 虽然你提到它们是一一对应的，但做一个长度检查更安全
        if len(original_ports_data) != len(port_types):
            print(f"警告: Item '{item_object.name}' 的 ports 和 port_layers 长度不匹配。将按最短长度处理。")
            # pass # zip 会自动按最短长度处理，或者你可以选择其他错误处理方式

        new_ports_list_for_item = []
        for port_coord_list_of_lists, port_type in zip(original_ports_data, port_types):
            # port_coord_list_of_lists 示例: [[14.5, 67.0], [130.5, 67.0], ...]
            # 将内部的坐标列表 [x, y] 转换为元组 (x, y)
            coordinates_as_tuples = [list(coord_pair) for coord_pair in port_coord_list_of_lists]

            new_port_dict = {
                'coordinates': coordinates_as_tuples,
                'type': port_type
            }
            new_ports_list_for_item.append(new_port_dict)

        # 直接修改 Item 对象的 ports 属性
        item_object.ports = new_ports_list_for_item

    # 函数修改了列表中的对象，可以不返回，或者返回修改后的列表
    return list_of_items


class RoutingScore_Interface():
    @staticmethod
    def Solve_Interface(items, area, links, old_links, start_time=None, max_time=None):
        """
        主流程函数，协调整个布线过程。
        """
        routing_local_start_time = time.time()
        print("--- 开始芯片布线算法 ---")

        # 1. 设置参数
        #config_id = extract_config_identifier(links_path)
        #params = setup_parameters(config_id)

        # 2. 加载数据
        # links, area, items = load_data(links_path, items_path)
        # print("===links:" , links, "===")
        # print("===area:" , area, "===")

        # start_time = datetime.now()
        items = transform_item_list(items)
        # item坐标需要根据area左下角点进行修正
        x = area[0][0]
        y = area[0][1]
        for item in items:
            for coor in item.boundary:
                coor[0] -= x
                coor[1] -= y
            for port in item.ports:
                for i, coor in enumerate(port['coordinates']):
                    port['coordinates'][i] = (coor[0] - x, coor[1] - y)

        area = list(area)
        for i, coor in enumerate(area):
            area[i] = (coor[0] - x, coor[1] - y)

        # 2.1 改为原系统接口 item需要改ports部分
        ## 根据link确定config_id，config_id概念换成connect_5.txt里面的link数，不再是原来的5
        config_id = len(old_links)
        params = setup_parameters(config_id)

        # 3. 初始化 RoutingScore 和布线数据
        rs = RoutingScore(items, area, links)
        # 返回: source_raw, target_raw, source_grid, target_grid, grid_list, weight_list, modulelist_Target, modulelist_Source, links
        return_vals = initialize_routing(rs, items, area, links, params)
        # if return_vals is None:
        #     sys.exit(1)

        # (source_raw_val, target_raw_val,source_grid_val, target_grid_val,grid_list_val,weight_list_val, modulelist_Target, modulelist_Source,
        #  extracted_links_from_return_vals) = return_vals  # 解包

        (
            source_13_raw, target_13_raw, source_list_13_grid, target_list_13_grid,
            grid_13_list, weight_13_list, modulelist_13_Target, modulelist_13_Source,
            links_13,  # 这是排序后的 links_13

            source_2_raw, target_2_raw, source_list_2_grid, target_list_2_grid,
            grid_2_list, weight_2_list, modulelist_2_Target, modulelist_2_Source,
            links_2  # 这是排序后的 links_2
        ) = return_vals

        # links = extracted_links_from_return_vals

        initial_data_13 = (
            source_13_raw, target_13_raw, source_list_13_grid, target_list_13_grid,
            grid_13_list, weight_13_list, modulelist_13_Target, modulelist_13_Source
        )

        initial_data_2 = (
            source_2_raw, target_2_raw, source_list_2_grid, target_list_2_grid,
            grid_2_list, weight_2_list, modulelist_2_Target, modulelist_2_Source
        )

        # 4. 运行 LAHC 优化
        # best_score只是优化过程的score
        # best_13_solution, best_13_score, best_fail_item_13_list, best_count_linked_items_13_num = run_lahc_optimization(rs, items, area, links_13, initial_data_13, params)
        # best_2_solution, best_2_score, best_fail_item_2_list, best_count_linked_items_2_num = run_lahc_optimization(rs, items, area, links_2, initial_data_2, params)
        if len(links_13) != 0:
            best_13_solution, best_13_score, best_return_vals_13 = run_lahc_optimization(rs, items, area, links_13,
                                                                                         initial_data_13, params,
                                                                                         old_links, start_time,
                                                                                         max_time)
            (items, items_area_list, connected_items_13, connected_items_area_list_13,
             success_area_size_13, success_path_length_13, SuccessRate_WiringConnection_13, current_solution_13,
             fail_item_list_13) = best_return_vals_13
        else:
            best_13_solution = []
            best_13_score = 0
            best_return_vals_13 = None
        if len(links_2) != 0:
            best_2_solution, best_2_score, best_return_vals_2 = run_lahc_optimization(rs, items, area, links_2,
                                                                                      initial_data_2, params,
                                                                                      old_links, start_time,
                                                                                      max_time)
            (items, items_area_list, connected_items_2, connected_items_area_list_2,
             success_area_size_2, success_path_length_2, SuccessRate_WiringConnection_2, current_solution_2,
             fail_item_list_2) = best_return_vals_2
        else:
            best_2_solution = []
            best_2_score = 0
            best_return_vals_2 = None
        # # 1. 合并两个列表
        # merged_list = fail_item_list_13 + fail_item_list_2
        #
        # # 2. 转换为集合去重，然后再转换回列表
        # best_fail_item_list = list(set(merged_list))
        # 5. 输出最终结果
        routing_local_end_time = time.time()
        elapsed_time = routing_local_end_time - routing_local_start_time
        # Todo: 注释

        # print(f"第13层最终最优分数 (Best Score): {(best_13_score):.2f}")
        # print(f"第2层最终最优分数 (Best Score): {(best_2_score):.2f}")
        # print(f"第13层漏连的item: {fail_item_list_13}")
        # print(f"第2层漏连的item: {fail_item_list_2}")
        # print(f"总漏连的item: {best_fail_item_list}")
        # print(f"总成功连接item数: {len(items) - len(best_fail_item_list)}")
        # print(f"总运行时间: {elapsed_time.total_seconds():.2f} 秒")

        # 6. 绘制最终结果图
        # plot_final_result(rs, best_13_solution, initial_data_13, area, items, params)
        # plot_final_result(rs, best_2_solution, initial_data_2, area, items, params)
        # 论文图片
        plot_final_result(rs, [], initial_data_13, area, items, params)
        plot_final_result(rs, [], initial_data_2, area, items, params)

        print("===本次布线得分 ", (best_13_score + best_2_score) / 2, " ===")
        print("===本次耗时  ", elapsed_time, " ===")
        print("--- 芯片布线算法结束 ---")

        # for item in items:
        #     print("布线结束item:", item.to_stringbyczg())
        return elapsed_time, (best_13_score + best_2_score) / 2





# --- 程序入口 ---
if __name__ == '__main__':
    pass

    # sec =  0
    # item = 5
    # count = 10
    # list_score = []
    # score = 0
    # for i in range(1, count+1):
    # # for i in range(1, 2):
    #     # 定义文件路径 (用户需要修改为实际路径)
    #     links_file = f'D:/本科/比赛/2022.09.01 EDA图像拼接/2024更新布线算法/data/EDA_DATA/connect/connect_file/connect_{item}.txt'
    #     item_file = f'D:/本科/比赛/2022.09.01 EDA图像拼接/2024更新布线算法/data/EDA_DATA/sample{item}/{item}-{i}/placement_info.txt'
    #     # links_file = f'D:/本科/比赛/2022.09.01 EDA图像拼接/2024更新布线算法/data/EDA_DATA/connect/connect_file/connect_{item}.txt'
    #     # item_file = f'D:/本科/比赛/2022.09.01 EDA图像拼接/2024更新布线算法/data/EDA_DATA/sample{item}/{item}-6/placement_info.txt'
    #
    #     # 调用主流程函数
    #     tmp_sec, tmp_score= Solve_Interface(links_file, item_file)
    #     sec += tmp_sec
    #     score += tmp_score
    #     list_score.append(tmp_score)
    #
    # print(f"===平均求解时间:{sec/count}===")
    # print(f"===得分list:{list_score}===")
    # print(f"===得分平均值:{score/count}===")
