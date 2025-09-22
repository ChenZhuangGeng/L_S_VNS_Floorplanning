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
from pylab import mpl

from src.classes.EdaClasses import RotateItem, Item
from src.function import ReadFileUtils


class TimeoutException(Exception):
    pass

class RoutingScore():
    def __init__(self, items, area, links):
        """
        构造函数 - 用于初始化 RoutingScore 实例。
        """
        self.items = items
        self.area = area
        self.links = links

    def run_for_getOneSolution(self, area, items, source_list_grid, target_list_grid, grid_list, weight_list, links, params, modulelist_Target, modulelist_Source, start_time, max_time):

        all_paths = []
        miss_items = []
        success_connect_count = 0
        for i in range(len(grid_list)):
            link = links[i]
            grid = grid_list[i]
            weight = weight_list[i]
            miss_items_temp = []

            for path in all_paths:
                grid = self.update_grid(grid, path)
                weight = self.update_weight(weight, path, params['grid_size'], params['affected_radius'], params['affected_value_for_path'],
                                          params['used_value'])

            grid, weights, paths = self.multi_source_region_search_V2(grid, weight, source_list_grid[i],
                                                                    target_list_grid[i], start_time, max_time)
            if len(paths) != len(source_list_grid[i]):
                miss_items_temp.extend(modulelist_Target[i])
                miss_items_temp.extend(modulelist_Source[i])
                paths = []
            else:
                success_connect_count += 1
            miss_items.append(miss_items_temp)
            all_paths.append(paths)

        # 对miss_items进行整理去重
        flat_list = [item for sublist in miss_items if sublist for item in sublist]
        miss_items = list(dict.fromkeys(flat_list))

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
                if len(path) < 2:
                    continue
                for i in range(len(path) - 1):
                    point1 = path[i]
                    point2 = path[i + 1]
                    success_path_length += abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

        SuccessRate_WiringConnection = success_connect_count / len(links)


        return items, items_area_list, connected_items, connected_items_area_list, success_area_size, success_path_length, SuccessRate_WiringConnection, all_paths, miss_items


    def convert_to_grid_coordinates(self, x, y, area_min, area_max, grid_size):
        raw_col = (x - area_min[0]) / grid_size
        raw_row = (area_max[1] - y) / grid_size

        col = math.floor(raw_col)
        row = math.floor(raw_row)
        return row, col



    def set_grid_weights(self, area, items, grid_size):
        area_min = area[0]
        area_max = area[2]
        width = area[2][0] - area[0][0]
        height = area[2][1] - area[0][1]
        print("width", width)
        print("height", height)

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
                    grid_x = np.round(area_min[0] + col * grid_size + grid_size / 2, decimals=6)
                    grid_y = np.round(area_max[1] - row * grid_size - grid_size / 2, decimals=6)

                    if poly_path.contains_point((grid_x, grid_y)):
                        grid[row, col] = 1

        return grid

    def set_grid_weights_for_source_target(self, area, items, grid_size, link, affected_radius, affected_value_for_item,
                                           used_value):
        area_min = area[0]
        area_max = area[2]
        width = area[2][0] - area[0][0]
        height = area[2][1] - area[0][1]

        num_cols = int(width / grid_size)
        num_rows = int(height / grid_size)

        # 创建网格
        grid = np.zeros((num_rows, num_cols))
        weight = [[1] * num_cols for _ in range(num_rows)]

        # 权重建立,还缺少线与线之间的间距，已完成
        for item in items:
            # 创建多边形路径
            poly_path = Path(item.boundary)
            # 创建版图的多边形
            item_polygon = Polygon(item.boundary)
            affected_polygon = item_polygon.buffer(affected_radius)

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

            for item_name, port_num in link.items():
                if item.name == item_name:
                    port_index = (int(port_num) - 1)
                    port = item.ports[port_index]

                    poly_path = Path(port['coordinates'])
                    for row in range(num_rows):
                        for col in range(num_cols):
                            grid_x = (area_min[0] + col * grid_size + grid_size / 2)
                            grid_y = (area_max[1] - row * grid_size - grid_size / 2)
                            if poly_path.contains_point((grid_x, grid_y)):
                                grid[row, col] = 2  # 标记端口

            end_time = datetime.now()

        return grid, weight


    def set_grid_weights_for_source_target_custom_corners(self, area, items, grid_size, link, affected_radius,
                                                          affected_value_for_item,
                                                          used_value, Flag):
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


        grid = np.zeros((num_rows, num_cols), dtype=int)
        default_weight = 1.0
        weight = [[default_weight] * num_cols for _ in range(num_rows)]

        grid_2 = copy.deepcopy(grid)
        weight_2 = copy.deepcopy(weight)

        if  Flag == 13:
            return grid_2, weight_2

        for item in items:
            item_boundary = item.boundary

            try:
                item_polygon_shapely = ShapelyPolygon(item_boundary)
                affected_polygon_shapely = item_polygon_shapely.buffer(affected_radius)
            except Exception as e:
                print(f"警告: 无法为 item {item.name} 创建或缓冲 Shapely 多边形: {e}")
                item_polygon_shapely = None
                affected_polygon_shapely = None
            minx, miny = 10000000000, 10000000000
            maxx, maxy = -10000000000, -10000000000

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

                    if self.is_point_in_polygon(point_coords, item_boundary):
                        grid[row, col] = 1
                        weight[row][col] = used_value
                    elif affected_polygon_shapely and affected_polygon_shapely.contains(point_for_shapely):
                        if grid[row, col] == 0:
                            weight[row][col] = affected_value_for_item

            for item_name, port_num in link.items():
                if item.name == item_name:

                    port_index = int(port_num) - 1
                    if 0 <= port_index < len(item.ports):
                        port = item.ports[port_index]
                        port_coordinates = port.get('coordinates')  # 获取端口边界坐标

                        minx, miny = 10000000000,10000000000
                        maxx, maxy = -10000000000, -10000000000

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

        return grid, weight # 对应13层


    def is_point_in_polygon(self, point, polygon_vertices):
        px, py = point
        num_vertices = len(polygon_vertices)
        is_inside = False

        if num_vertices < 3:
            return False

        p1x, p1y = polygon_vertices[0]
        for i in range(num_vertices + 1):
            p2x, p2y = polygon_vertices[i % num_vertices]

            if min(p1y, p2y) <= py < max(p1y, p2y):
                if p2y - p1y != 0:
                    x_intersection = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if x_intersection >= px:
                        is_inside = not is_inside
            elif p1y == p2y == py:
                if min(p1x, p2x) <= px <= max(p1x, p2x):
                    p1x, p1y = p2x, p2y

        for i in range(num_vertices):
            if polygon_vertices[i] == point:
                return True

        return is_inside

    def multi_source_region_search_V2(self, grid, weights, source_area_Out, target_area_Out, start_time, max_time):
        source_area_in = source_area_Out.copy()
        target_area_in = target_area_Out.copy()
        rows, cols = len(grid), len(grid[0])

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

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
            dist = [[float('inf')] * cols for _ in range(rows)]
            parent = [[None] * cols for _ in range(rows)]
            pq = []
            for x, y in source_area:
                dist[x][y] = 0
                heapq.heappush(pq, (0, x, y))  # (距离, x, y)


            while pq:
                # 在每次从队列取元素时检查时间，这是最高频的操作，控制最精确
                if start_time and max_time and (time.time() - start_time >= max_time):
                    raise TimeoutException("寻路超时！")

                current_dist, x, y = heapq.heappop(pq)

                if [x, y] in target_area:
                    return (x, y), parent

                if current_dist > dist[x][y]:
                    continue

                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != 1:
                        new_dist = dist[x][y] + weights[nx][ny]
                        if new_dist < dist[nx][ny]:
                            dist[nx][ny] = new_dist
                            parent[nx][ny] = (x, y)
                            heapq.heappush(pq, (new_dist, nx, ny))
            return None, None

        # 存储路径
        all_paths = []
        all_connected = []
        used_source = []
        used_target = []
        while len(source_area_in) >= 1:
            nearest_target, parent = dijkstra_search(source_area_in, target_area_in, start_time, max_time)
            if (nearest_target, parent) == (None, None):
                return grid, weights, all_paths
            if nearest_target:
                path = find_path(nearest_target, source_area_in, parent)

                used_source.append(path[0])
                used_target.append(nearest_target)
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
                grid[x][y] = 1
        return grid

    def update_weight(self, weight, paths, grid_size, affected_radius, affected_value_for_path, used_value):
        affected_range = (int)(affected_radius / grid_size)

        for path in paths:
            for x, y in path:
                weight[x][y] = used_value
                for dx in range(-affected_range, affected_range + 1):
                    for dy in range(-affected_range, affected_range + 1):
                        nx, ny = x + dx, y + dy

                        if 0 <= nx < len(weight) and 0 <= ny < len(weight[0]):
                            if weight[nx][ny] != used_value:
                                weight[nx][ny] = affected_value_for_path
        return weight


class DataUtils():
    @staticmethod
    def get_max_item_number_from_links_regex(links_list):
        max_number = 0
        if not links_list:
            return 0

        pattern = re.compile(r'^M(\d+)$')

        for link_dict in links_list:
            for key in link_dict.keys():
                match = pattern.match(key)
                if match:
                    number_str = match.group(1)
                    number = int(number_str)
                    if number > max_number:
                        max_number = number
        return max_number
    @staticmethod
    def filter_invalid_links(links_list, items_list):
        valid_item_names = set()
        for item_obj in items_list:
            valid_item_names.add(item_obj.name)

        filtered_links = []
        for link_dict in links_list:
            all_keys_valid = True
            for key_name in link_dict.keys():
                if key_name not in valid_item_names:
                    all_keys_valid = False
                    break

            if all_keys_valid:
                filtered_links.append(link_dict)

        return filtered_links
    @staticmethod
    def calculate_overall_bounding_box_area(connected_items, list_of_original_path_groups) -> float:

        min_x_overall = math.inf
        min_y_overall = math.inf
        max_x_overall = -math.inf
        max_y_overall = -math.inf

        points_found = False

        for item_idx, item in enumerate(connected_items):
            boundary = []
            if hasattr(item, 'boundary'):
                item_boundary_attr = getattr(item, 'boundary')
                if isinstance(item_boundary_attr, list):
                    boundary = item_boundary_attr
            elif isinstance(item, dict) and 'boundary' in item:
                item_boundary_dict_val = item['boundary']
                if isinstance(item_boundary_dict_val, list):
                    boundary = item_boundary_dict_val

            if not boundary:
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
                        pass
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
                                        pass

        if not points_found:
            print("信息: 未找到任何有效坐标点，无法计算外包矩形面积。")
            return 0.0

        width = max_x_overall - min_x_overall
        height = max_y_overall - min_y_overall
        if width < 0 or height < 0:
            print(f"警告: 计算出的宽度或高度为负 (宽:{width}, 高:{height})。可能所有点共线或数据问题。返回0面积。")
            return 0.0

        return width * height


    @staticmethod
    def _convert_point_from_grid(
            grid_row,
            grid_col,
            area_min_tuple,
            area_max_tuple,
            grid_size
    ):
        min_x = area_min_tuple[0]
        max_y_for_inversion = area_max_tuple[1]

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
        all_original_path_groups = []
        if not list_of_grid_path_groups:
            return []

        for i, grid_path_group in enumerate(list_of_grid_path_groups):
            current_original_path_group = []
            if not isinstance(grid_path_group, list):
                print(
                    f"警告: 最外层列表(路径组列表)中索引 {i} 的元素 '{grid_path_group}' 不是列表类型，将视为空路径组处理。")
                all_original_path_groups.append(current_original_path_group)
                continue
            for j, grid_segment in enumerate(grid_path_group):
                current_original_segment = []
                if not isinstance(grid_segment, list):
                    print(
                        f"警告: 路径组索引 {i} 中，段索引 {j} 的元素 '{grid_segment}' 不是列表类型（段），将视为空段处理。")
                    current_original_path_group.append(current_original_segment)
                    continue

                if not grid_segment:
                    current_original_path_group.append(current_original_segment)
                    continue

                for k, grid_point in enumerate(grid_segment):
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
        if not vertices or not isinstance(vertices, list):
            # print("警告: 提供的顶点列表无效或为空。")
            return 0.0

        n = len(vertices)
        if n < 3:
            # print(f"警告: 至少需要3个顶点来构成多边形，当前只有 {n} 个。")
            return 0.0

        for i, point in enumerate(vertices):
            if not (isinstance(point, list) and len(point) == 2 and
                    all(isinstance(coord, (int, float)) for coord in point)):
                # print(f"警告: 顶点列表中的索引 {i} 处的点 '{point}' 格式无效。面积计为0。")
                return 0.0

        area_sum = 0.0
        for i in range(n):
            x_i = vertices[i][0]
            y_i = vertices[i][1]

            x_i_plus_1 = vertices[(i + 1) % n][0]
            y_i_plus_1 = vertices[(i + 1) % n][1]
            area_sum += (x_i * y_i_plus_1) - (x_i_plus_1 * y_i)

        return abs(area_sum) / 2.0

    @staticmethod
    def calculate_item_areas(items: list) -> list:
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

            area = DataUtils._calculate_polygon_area_shoelace(boundary_points)
            if area == 0.0 and (not boundary_points or len(boundary_points) < 3):
                pass

            areas_list.append(area)

        return areas_list

    @staticmethod
    def find_connected_items(items: list, miss_items_names: list) -> list:
        connected_items_list = []
        miss_items_set = set(miss_items_names)

        for item in items:
            if hasattr(item, 'name'):
                item_name = item.name
                if item_name not in miss_items_set:
                    connected_items_list.append(item)
            else:
                print(f"警告: 项目 {item} (类型: {type(item)}) 没有 'name' 属性，将被忽略。")

        return connected_items_list

    @staticmethod
    def find_missing_items_in_paths(target_names, target_coords, source_names, source_coords, paths):
        missing_items = []
        target_coord_map = {tuple(coord): name for coord, name in zip(target_coords, target_names)}
        source_coord_map = {tuple(coord): name for coord, name in zip(source_coords, source_names)}

        actual_start_points = set()
        actual_end_points = set()

        if paths:
            for path in paths:
                if path and isinstance(path, collections.abc.Sequence) and len(path) > 0:
                    try:
                        actual_start_points.add(path[0])
                        actual_end_points.add(path[-1])
                    except TypeError:
                        if isinstance(path[0], list):
                            actual_start_points.add(tuple(path[0]))
                        if isinstance(path[-1], list):
                            actual_end_points.add(tuple(path[-1]))
        for coord, name in target_coord_map.items():
            if coord not in actual_end_points:
                missing_items.append(name)

        for coord, name in source_coord_map.items():
            if coord not in actual_start_points:
                missing_items.append(name)

        return missing_items

    @staticmethod
    def sort_lists_based_on_first(list_to_sort_by, list_to_follow):

        assert len(list_to_sort_by) == len(list_to_follow), \
            "输入列表 list_to_sort_by 和 list_to_follow 的长度必须相同！"

        if not list_to_sort_by:
            return [], []

        combined_list = list(zip(list_to_sort_by, list_to_follow))

        sorted_combined_list = sorted(combined_list, key=lambda item: item[0])
        if not sorted_combined_list:
            return [], []
        sorted_list1, sorted_list2 = map(list, zip(*sorted_combined_list))

        return sorted_list1, sorted_list2

    @staticmethod
    def calculate_total_pairwise_manhattan_distance(points_list):
        n = len(points_list)
        if n < 2:
            return 0.0

        total_distance = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                p1 = points_list[i]
                p2 = points_list[j]
                distance = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
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
        area = None
        links = None
        name_list = None
        rotate_items = []

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
        if not rotate_items: print("没有 RotateItem 可供绘制。"); return
        fig, ax = plt.subplots(figsize=(10, 8))
        if area:
             area_patch = patches.Polygon(area, closed=True, edgecolor='gray', facecolor='none', linestyle='--', lw=1)
             ax.add_patch(area_patch)
             min_x, max_x = min(p[0] for p in area) - 20, max(p[0] for p in area) + 20
             min_y, max_y = min(p[1] for p in area) - 20, max(p[1] for p in area) + 20
        else:
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

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.tick_params(axis='both', which='both', length=0)

        x_min, x_max = ax.get_xlim()
        y_max, y_min = ax.get_ylim()

        from matplotlib.patches import Rectangle
        rect_width = x_max - x_min
        rect_height = y_max - y_min

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
        if not lists: return []
        try:
            first_len = len(lists[0])
            if not all(len(lst) == first_len for lst in lists[1:]):
                 raise ValueError("所有列表长度必须相同！")
        except IndexError:
             return [[] for _ in lists]


        num_items = first_len
        if num_items == 0: return [[] for _ in lists]

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
        source_coords = [tuple(item) for sublist in source_list_grid for item in sublist]
        target_coords = [tuple(item) for sublist in target_list_grid for item in sublist]

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

        all_path_coords = [coord_tuple for path in all_paths for sub_path in path for coord_tuple in sub_path]

        for coord in source_coords:
            if coord not in all_path_coords:
                missing_coords['source']['coords'].append(coord)
                missing_coords['source']['isMissing'] = True
                missing_coords['source']['count'] += 1

        for coord in target_coords:
            if coord not in all_path_coords:
                missing_coords['target']['coords'].append(coord)
                missing_coords['target']['isMissing'] = True
                missing_coords['target']['count'] += 1

        return missing_coords

    def evaluate_paths(all_paths, source_list_grid, target_list_grid):
        missing_coords = DataUtils.check_missing_coordinates(all_paths, source_list_grid, target_list_grid)
        if missing_coords['source']['isMissing'] or missing_coords['target']['isMissing']:
            c = missing_coords['source']['count'] + missing_coords['target']['count']
            return False, 0, c

        total_length = 0
        for paths in all_paths:
            for path in paths:
                if len(path) < 2:
                    continue
                for i in range(len(path) - 1):
                    point1 = path[i]
                    point2 = path[i + 1]
                    if not (isinstance(point1, tuple) and isinstance(point2, tuple)):
                        raise ValueError("Each path must consist of coordinate tuples (x, y).")
                    total_length += abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

        return True, total_length, 0
    @staticmethod
    def getScore(items, area, path):
        print("items:",items[0])
        print("area:", area)
        print("path:", path)
        print("end")
        pass




def load_data(links_path, items_path):
    print(f"加载连接文件: {links_path}")
    links = ReadFileUtils.parse_links_file(links_path)
    print(f"加载布局文件: {items_path}")
    area, items = ReadFileUtils.parse_items_file(items_path)
    print("area", area)
    print("items", items)
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

def extract_config_identifier(links_path):
    try:
        filename = os.path.basename(links_path) # 例如：connect_40.txt
        match = re.search(r'connect_(\d+)', filename)
        if match:
            return int(match.group(1))
        else:
            print(f"警告：无法从 '{filename}' 中提取标识符。将使用默认参数。")
            return None
    except Exception as e:
        print(f"从 '{links_path}' 提取标识符时出错：{e}。将使用默认参数。")
        return None

def setup_parameters(config_identifier=None): # 添加了 config_identifier 参数
    grid_size = 5
    params = {
        'grid_size': grid_size,
        'port_boundary_index': 0,
        'affected_radius': 2 * grid_size,
        'affected_value_for_item': 2000.0,
        'affected_value_for_path': 200.0,
        'used_value': 10000.0,
        'history_length': 15,
        'max_iterations': 50,
        'penalty_factor': 100000.0
    }

    # 根据 config_identifier 调整参数
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
        if config_identifier is not None:
            print(f"警告：未找到标识符 '{config_identifier}' 的特定参数。将使用默认/基础参数。")
        else:
            print("使用默认/基础参数。")

    if params['grid_size'] != grid_size:
        params['affected_radius'] = 2 * params['grid_size']
    return params


def initialize_routing(rs, items, area, links, params):
    start_time = datetime.now()

    links_2 = []
    links_13 = []

    for item in links:
        if item and all(value == '2' for value in item.values()):
            links_2.append(item)
        else:
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
    grid_13_list = []
    weight_13_list = []
    for i in range(len(links_13)):
        grid, weight = rs.set_grid_weights_for_source_target_custom_corners(area, items, params['grid_size'], links_13,
                                                                              params['affected_radius'],
                                                                              params['affected_value_for_item'],
                                                                              params['used_value'], Flag=13)
        grid_13_list.append(grid)
        weight_13_list.append(weight)

    link_dist_list = []
    for connected_ports in connected_13_ports:
        gravity_ports_list = []
        for port in connected_ports:
            gravity = DataUtils.calculate_polygon_centroid(port['port_coordinates'])
            gravity_ports_list.append((gravity))

        dist = DataUtils.calculate_total_pairwise_manhattan_distance(gravity_ports_list)
        link_dist_list.append(dist)
    link_dist_list, links_13 = DataUtils.sort_lists_based_on_first(link_dist_list, links_13)

    link_dist_list = []
    for connected_ports in connected_2_ports:
        gravity_ports_list = []
        for port in connected_ports:
            gravity = DataUtils.calculate_polygon_centroid(port['port_coordinates'])
            gravity_ports_list.append((gravity))

        dist = DataUtils.calculate_total_pairwise_manhattan_distance(gravity_ports_list)
        link_dist_list.append(dist)
    link_dist_list, links_2 = DataUtils.sort_lists_based_on_first(link_dist_list, links_2)

    end_time = datetime.now()
    elapsed_time = end_time - start_time

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
    return (
        source_13_raw, target_13_raw, source_list_13_grid, target_list_13_grid,
        grid_13_list, weight_13_list, modulelist_13_Target, modulelist_13_Source,
        links_13,

        source_2_raw, target_2_raw, source_list_2_grid, target_list_2_grid,
        grid_2_list, weight_2_list, modulelist_2_Target, modulelist_2_Source,
        links_2
    )

def calculate_score(solution, source_grid, target_grid, params):
    valid, total_length, missing_count = DataUtils.evaluate_paths(solution, source_grid, target_grid)
    if valid:
        return total_length
    else:
        penalty_factor = params.get('penalty_factor', 100000.0)
        if missing_count > 0:
             return total_length + missing_count * penalty_factor
        else:
             return float('inf')


def calculate_chip_score(
        items,
        items_area_list,
        connected_items,
        connected_items_area_list,
        success_area_size,
        success_path_length,
        SuccessRate_WiringConnection,
        old_links,
        k_wl_param=0.01,
):
    total_module_count = DataUtils().get_max_item_number_from_links_regex(old_links)
    successful_module_count = len(connected_items)

    if total_module_count == 0:
        return 0.0

    total_module_area = sum(items_area_list)
    successful_module_area = sum(connected_items_area_list)
    module_utilization_rate = successful_module_count / total_module_count

    module_area_utilization_rate = 0.0
    if total_module_area > 0:
        module_area_utilization_rate = successful_module_area / total_module_area

    if total_module_count < 25:
        base_routing_score = 50
    elif 25 <= total_module_count <= 40:
        base_routing_score = 70
    else:
        base_routing_score = 90

    all_modules_successful = (successful_module_count == total_module_count)

    if all_modules_successful:
        score_part1 = float(base_routing_score)
    else:
        score_part1 = base_routing_score * module_utilization_rate * module_area_utilization_rate

    bonus_scores_eligible = False
    if total_module_count < 25:
        if successful_module_count >= total_module_count - 2:
            bonus_scores_eligible = True
    elif 25 <= total_module_count <= 50:
        if module_utilization_rate >= 0.90:
            bonus_scores_eligible = True
    else:
        if module_utilization_rate >= 0.85:
            bonus_scores_eligible = True

    score_part1_2 = 0.0
    if bonus_scores_eligible:
        if success_area_size > 0 and successful_module_area >= 0:
            density = successful_module_area / success_area_size
            score_part1_2 = 10.0 * density
            score_part1_2 = min(score_part1_2, 10.0)

    score_part1_3 = 0.0
    if bonus_scores_eligible:
        avg_wl_per_module = 0.0
        if successful_module_count > 0 and success_path_length >= 0:
            avg_wl_per_module = success_path_length / successful_module_count

        wirelength_penalty_factor = math.exp(-k_wl_param * avg_wl_per_module)

        score_part1_3 = 10.0 * SuccessRate_WiringConnection * wirelength_penalty_factor

    total_score = score_part1 + score_part1_2 + score_part1_3
    return total_score

def run_lahc_optimization(rs, items, area, links, initial_data, params, old_links, start_time, max_time):
    if initial_data is None:
        print("错误: 初始数据无效，无法开始 LAHC。")
        return [], float('inf')

    source_grid, target_grid, grid_list, weight_list, modulelist_Target, modulelist_Source = initial_data[2:] # 解包需要的部分

    initial_input_data = [copy.deepcopy(source_grid), copy.deepcopy(target_grid),
                          copy.deepcopy(grid_list), copy.deepcopy(weight_list), copy.deepcopy(links)]

    return_vals = rs.run_for_getOneSolution(
        area, items, copy.deepcopy(initial_input_data[0]), copy.deepcopy(initial_input_data[1]), copy.deepcopy(initial_input_data[2]),
        copy.deepcopy(initial_input_data[3]), copy.deepcopy(initial_input_data[4]), params, modulelist_Target, modulelist_Source, start_time, max_time
    )
    (items, items_area_list, connected_items, connected_items_area_list,
     success_area_size, success_path_length, SuccessRate_WiringConnection, current_solution, fail_item_list) = return_vals
    current_score = calculate_chip_score(items, items_area_list, connected_items, connected_items_area_list,
     success_area_size, success_path_length, SuccessRate_WiringConnection, old_links)


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

        # 评估新解
        new_score = calculate_chip_score(items, items_area_list, connected_items, connected_items_area_list,
                                             success_area_size, success_path_length, SuccessRate_WiringConnection, old_links)

        # LAHC 接受准则
        historical_score_idx = (history_index - 1 + params['history_length']) % params['history_length']
        historical_score = history[historical_score_idx]

        if new_score > current_score or new_score > historical_score:
            current_solution = copy.deepcopy(new_solution)
            current_score = new_score
            current_return_vals = copy.deepcopy(return_vals)
            accepted_input_data = perturbed_lists # 保存接受解的输入数据

            if current_score > best_score:
                best_solution = copy.deepcopy(current_solution)
                best_score = current_score
                best_return_vals = copy.deepcopy(current_return_vals)
                # print(f"迭代 {iteration + 1}/{params['max_iterations']}: *新最优解* Score = {best_score:.2f} (对比历史: {historical_score:.2f})")
            else:
                #print(f"迭代 {iteration + 1}/{params['max_iterations']}: 接受新解   Score = {current_score:.2f} (对比历史: {historical_score:.2f})")
                pass
        else:
            #print(f"迭代 {iteration + 1}/{params['max_iterations']}: 拒绝新解   Score = {new_score:.2f} (当前: {current_score:.2f}, 历史: {historical_score:.2f})")
            pass


        # 更新历史记录 (使用当前接受的分数)
        history[history_index] = current_score
        history_index = (history_index + 1) % params['history_length']

    return best_solution, best_score, best_return_vals

def plot_final_result(rs, best_solution, initial_data, area, items, params):
    if initial_data is None:
        print("错误: 初始数据无效，无法绘制结果。")
        return

    source_grid, target_grid = initial_data[2:4]

    try:
        # 创建用于绘图的基础网格 (只包含障碍)
        grid_raw_for_plot = rs.set_grid_weights(area, items, params['grid_size'])
        if grid_raw_for_plot is None or grid_raw_for_plot.size == 0:
             print("错误：无法生成用于绘图的基础网格。")
             return

        rows, cols = grid_raw_for_plot.shape
        all_port_coords = set(tuple(coord) for group in source_grid for coord in group)
        all_port_coords.update(set(tuple(coord) for group in target_grid for coord in group))

        for r, c in all_port_coords:
             if 0 <= r < rows and 0 <= c < cols and grid_raw_for_plot[r,c] != 1:
                 grid_raw_for_plot[r,c] = 2

        DataUtils.plot_grid_with_path(grid_raw_for_plot, best_solution, source_grid, target_grid, params)
    except Exception as e:
        print(f"绘制最终结果时出错: {e}")



def transform_item_list(list_of_items):
    for item_object in list_of_items:
        if not hasattr(item_object, 'ports') or not hasattr(item_object, 'port_layers'):
            print(f"警告: Item '{item_object.name}' 缺少 'ports' 或 'port_layers' 属性。")
            continue

        original_ports_data = item_object.ports
        port_types = item_object.port_layers
        item_object.boundary = [list(coord_pair) for coord_pair in item_object.boundary]
        if not isinstance(original_ports_data, list) or not isinstance(port_types, list):
            print(f"警告: Item '{item_object.name}' 的 'ports' 或 'port_layers' 不是列表类型。")
            continue

        if len(original_ports_data) != len(port_types):
            print(f"警告: Item '{item_object.name}' 的 ports 和 port_layers 长度不匹配。将按最短长度处理。")

        new_ports_list_for_item = []
        for port_coord_list_of_lists, port_type in zip(original_ports_data, port_types):
            coordinates_as_tuples = [list(coord_pair) for coord_pair in port_coord_list_of_lists]

            new_port_dict = {
                'coordinates': coordinates_as_tuples,
                'type': port_type
            }
            new_ports_list_for_item.append(new_port_dict)

        item_object.ports = new_ports_list_for_item

    return list_of_items


class RoutingScore_Interface():
    @staticmethod
    def Solve_Interface(items, area, links, old_links, start_time=None, max_time=None):
        routing_local_start_time = time.time()

        items = transform_item_list(items)
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

        config_id = len(old_links)
        params = setup_parameters(config_id)

        rs = RoutingScore(items, area, links)
        return_vals = initialize_routing(rs, items, area, links, params)

        (
            source_13_raw, target_13_raw, source_list_13_grid, target_list_13_grid,
            grid_13_list, weight_13_list, modulelist_13_Target, modulelist_13_Source,
            links_13,

            source_2_raw, target_2_raw, source_list_2_grid, target_list_2_grid,
            grid_2_list, weight_2_list, modulelist_2_Target, modulelist_2_Source,
            links_2
        ) = return_vals


        initial_data_13 = (
            source_13_raw, target_13_raw, source_list_13_grid, target_list_13_grid,
            grid_13_list, weight_13_list, modulelist_13_Target, modulelist_13_Source
        )

        initial_data_2 = (
            source_2_raw, target_2_raw, source_list_2_grid, target_list_2_grid,
            grid_2_list, weight_2_list, modulelist_2_Target, modulelist_2_Source
        )

        # 4. 运行 LAHC 优化
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

        routing_local_end_time = time.time()
        elapsed_time = routing_local_end_time - routing_local_start_time

        print("===本次布线得分 ", (best_13_score + best_2_score) / 2, " ===")
        print("===本次耗时  ", elapsed_time, " ===")

        return elapsed_time, (best_13_score + best_2_score) / 2
