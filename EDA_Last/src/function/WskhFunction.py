import linecache
import math
import os
import torch
from torch.utils.data import Dataset

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from decimal import Decimal

from torch_geometric.nn import global_mean_pool

from src.function import ReadFileUtils


def calc_envelope_rectangle_w_h(boundary):
    min_x = None
    max_x = None
    min_y = None
    max_y = None
    for x, y in boundary:
        if min_x is None:
            min_x = x
            max_x = x
            min_y = y
            max_y = y
        else:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
    return max_x - min_x, max_y - min_y


def calc_area(w, h, boundary):
    boundary = boundary.copy()
    min_x = min_y = None
    for p in boundary:
        if min_x is None or min_x > p[0]:
            min_x = p[0]

        if min_y is None or min_y > p[1]:
            min_y = p[1]
    boundary = [[p[0] - min_x, p[1] - min_y] for p in boundary]

    area = w * h
    l = len(boundary)
    for index in range(l):
        x, y = boundary[index]
        if x != 0 and y != 0 and x != w and y != h:
            x_f, y_f = boundary[index - 1 if index - 1 >= 0 else l - 1]
            x_b, y_b = boundary[index + 1 if index + 1 < l else 0]
            area -= ((abs(x - x_f) + abs(y - y_f)) * (abs(x - x_b) + abs(y - y_b)))
    return area


def calc_horizontal_line_list(boundary):
    horizontal_line_list = []
    l = len(boundary)
    for index in range(l):
        x, y = boundary[index]
        x_f, y_f = boundary[index - 1 if index - 1 >= 0 else l - 1]
        if y_f == y:
            if x < x_f:
                horizontal_line_list.append((x, y, x_f - x))
                continue
        x_b, y_b = boundary[index + 1 if index + 1 < l else 0]
        if y_b == y:
            if x < x_b:
                horizontal_line_list.append((x, y, x_b - x))
    return horizontal_line_list


def calc_vertical_line_list(boundary):
    vertical_line_list = []
    l = len(boundary)
    for index in range(l):
        x, y = boundary[index]
        x_f, y_f = boundary[index - 1 if index - 1 >= 0 else l - 1]
        if x_f == x:
            if y < y_f:
                vertical_line_list.append((x, y, y_f - y))
                continue
        x_b, y_b = boundary[index + 1 if index + 1 < l else 0]
        if x_b == x:
            if y < y_b:
                vertical_line_list.append((x, y, y_b - y))
    return vertical_line_list


def calc_add_sky_line_list(horizontal_line_list):
    add_sky_line_list = []
    for i in range(len(horizontal_line_list)):
        x_i, y_i, l_i = horizontal_line_list[i]
        b = True
        for j in range(len(horizontal_line_list)):
            x_j, y_j, l_j = horizontal_line_list[j]
            if i != j and y_j > y_i and line_is_intersect((x_i, x_i + l_i), (x_j, x_j + l_j)):
                b = False
                break
        if b:
            add_sky_line_list.append((x_i, y_i, l_i))
    return add_sky_line_list


def calc_aim_point_list(boundary):

    aim_point_list = [(0, 0)]
    min_x = None
    for x, y in boundary:
        if y == 0:
            if min_x is None or min_x > x:
                min_x = x
    if min_x != 0:
        aim_point_list.append((min_x, 0))
    return aim_point_list


def line_is_intersect(line1, line2):
    real_len = max(line1[0], line1[1], line2[0], line2[1]) - min(line1[0], line1[1], line2[0], line2[1])
    ideal_len = line1[1] - line1[0] + line2[1] - line2[0]
    return real_len < ideal_len


def is_overlap(rotate_item_1, left_bottom_point_1, rotate_item_2, left_bottom_point_2):
    for rect1 in rotate_item_1.rect_list:
        for rect2 in rotate_item_2.rect_list:
            if rect_is_overlap(rect1[2], rect1[3],
                               (rect1[0] + left_bottom_point_1[0], rect1[1] + left_bottom_point_1[1]), rect2[2],
                               rect2[3],
                               (rect2[0] + left_bottom_point_2[0], rect2[1] + left_bottom_point_2[1])) is True:
                return True
    return False


def rect_is_overlap(w, h, left_bottom_point, w2, h2, left_bottom_point2):
    if left_bottom_point2[0] + w2 <= left_bottom_point[0]:
        return False
    elif left_bottom_point2[0] >= left_bottom_point[0] + w:
        return False
    elif left_bottom_point2[1] + h2 <= left_bottom_point[1]:
        return False
    elif left_bottom_point2[1] >= h2 + left_bottom_point[1]:
        return False
    else:
        return True


def get_distance(point1, point2):
    diff = [(point1[x] - point2[x]) for x in range(2)]
    dist = math.sqrt(diff[0] ** 2 + diff[1] ** 2)
    return dist


def zoom_in_or_zoom_out_boundary(points, L):
    gravity_point = get_gravity_point(points)
    if len(points) <= 2 or not gravity_point:
        return list()

    length = len(points)
    normal_vector = list()
    for i in range(length):
        vector_x = points[(i + 1) % length][0] - points[i][0]
        vector_y = points[(i + 1) % length][1] - points[i][1]
        normal_vector_x = vector_x / math.sqrt(vector_x ** 2 + vector_y ** 2)
        normal_vector_y = vector_y / math.sqrt(vector_x ** 2 + vector_y ** 2)
        normal_vector.append([normal_vector_x, normal_vector_y])

    theta = list()
    for i in range(length):
        x1 = normal_vector[i][0]
        y1 = normal_vector[i][1]
        x2 = normal_vector[(i - 1) % len(points)][0]
        y2 = normal_vector[(i - 1) % len(points)][1]
        sin_theta = abs(x1 * y2 - x2 * y1)
        theta.append(sin_theta)

    new_points = list()
    min_x = min_y = None
    for i in range(length):
        point = points[i]
        x1 = -normal_vector[(i - 1) % len(points)][0]
        y1 = -normal_vector[(i - 1) % len(points)][1]
        x2 = normal_vector[i][0]
        y2 = normal_vector[i][1]
        add_x = L / theta[i] * (x1 + x2)
        add_y = L / theta[i] * (y1 + y2)
        new_point_x = point[0] + add_x
        new_point_y = point[1] + add_y
        new_point = [new_point_x, new_point_y]
        if get_distance(new_point, gravity_point) > get_distance(point, gravity_point):
            new_point[0] -= 2 * add_x
            new_point[1] -= 2 * add_y
        if min_x is None:
            min_x = new_point[0]
        else:
            min_x = min(min_x, new_point[0])
        if min_y is None:
            min_y = new_point[1]
        else:
            min_y = min(min_y, new_point[1])
        new_points.append(new_point)
    for i in range(len(new_points)):
        new_points[i] = [new_points[i][0] - min_x, new_points[i][1] - min_y]
    return new_points


def get_gravity_point(points):
    if len(points) <= 2:
        return list()

    area = Decimal(0.0)
    x, y = Decimal(0.0), Decimal(0.0)
    for i in range(len(points)):
        lng = Decimal(points[i][0])
        lat = Decimal(points[i][1])
        nextlng = Decimal(points[i - 1][0])
        nextlat = Decimal(points[i - 1][1])

        tmp_area = (nextlng * lat - nextlat * lng) / Decimal(2.0)
        area += tmp_area
        x += tmp_area * (lng + nextlng) / Decimal(3.0)
        y += tmp_area * (lat + nextlat) / Decimal(3.0)
    x = x / area
    y = y / area
    return [float(x), float(y)]




def zoom_use_exhaustive_method(rotate_item, dis):
    if len(rotate_item.boundary) == 4:
        w, h = rotate_item.w, rotate_item.h
        return [[0, 0], [w + dis, 0], [w + dis, h + dis], [0, h + dis]]
    elif len(rotate_item.boundary) == 6:
        if rotate_item.type == '0':
            w, h, w1, w2, h1, h2 = rotate_item.w, rotate_item.h, rotate_item.w1, rotate_item.w2, rotate_item.h1, rotate_item.h2
            return [[0, 0], [w + dis, 0], [w + dis, h2 + dis], [w1 + dis, h2 + dis], [w1 + dis, h + dis], [0, h + dis]]
        elif rotate_item.type == '90':
            w, h, w1, w2, h1, h2 = rotate_item.w, rotate_item.h, rotate_item.w1, rotate_item.w2, rotate_item.h1, rotate_item.h2
            return [[0, 0], [w1 + dis, 0], [w1 + dis, h1], [w + dis, h1], [w + dis, h + dis], [0, h + dis]]
        elif rotate_item.type == '180':
            w, h, w1, w2, h1, h2 = rotate_item.w, rotate_item.h, rotate_item.w1, rotate_item.w2, rotate_item.h1, rotate_item.h2
            return [[w1, 0], [w1, h2], [0, h2], [0, h + dis], [w + dis, h + dis], [w + dis, 0]]
        elif rotate_item.type == '270':
            w, h, w1, w2, h1, h2 = rotate_item.w, rotate_item.h, rotate_item.w1, rotate_item.w2, rotate_item.h1, rotate_item.h2
            return [[0, 0], [0, h1 + dis], [w1, h1 + dis], [w1, h + dis], [w + dis, h + dis], [w + dis, 0]]
    elif len(rotate_item.boundary) == 8:
        if rotate_item.type == '0':
            w, h, w1, w2, w3, h1, h2, h3, h4 = rotate_item.w, rotate_item.h, rotate_item.w1, rotate_item.w2, rotate_item.w3, rotate_item.h1, rotate_item.h2, rotate_item.h3, rotate_item.h4
            return [[0, 0], [0, h1 + dis], [w1, h1 + dis], [w1, h + dis], [w1 + w2 + dis, h + dis],
                    [w1 + w2 + dis, h4 + dis], [w + dis, h4 + dis], [w + dis, 0]]
        elif rotate_item.type == '90':
            w, h, w1, w2, w3, w4, h1, h2, h3 = rotate_item.w, rotate_item.h, rotate_item.w1, rotate_item.w2, rotate_item.w3, rotate_item.w4, rotate_item.h1, rotate_item.h2, rotate_item.h3
            return [[0, 0], [0, h + dis], [w4 + dis, h + dis], [w4 + dis, h2 + h3 + dis], [w + dis, h2 + h3 + dis],
                    [w + dis, h3], [w1 + dis, h3], [w1 + dis, 0]]
        elif rotate_item.type == '180':
            w, h, w1, w2, w3, h1, h2, h3, h4 = rotate_item.w, rotate_item.h, rotate_item.w1, rotate_item.w2, rotate_item.w3, rotate_item.h1, rotate_item.h2, rotate_item.h3, rotate_item.h4
            return [[w1, 0], [w1, h2], [0, h2], [0, h + dis], [w + dis, h + dis], [w + dis, h3], [w1 + w2 + dis, h3],
                    [w1 + w2 + dis, 0]]
        elif rotate_item.type == '270':
            w, h, w1, w2, w3, w4, h1, h2, h3 = rotate_item.w, rotate_item.h, rotate_item.w1, rotate_item.w2, rotate_item.w3, rotate_item.w4, rotate_item.h1, rotate_item.h2, rotate_item.h3
            return [[w2, 0], [w2, h1], [0, h1], [0, h1 + h2 + dis], [w3, h1 + h2 + dis], [w3, h + dis],
                    [w + dis, h + dis],
                    [w + dis, 0]]
    return rotate_item.boundary


def get_left_bottom_point(boundary):
    min_x, min_y = None, None
    for p in boundary:
        if min_x is None or min_x > p[0]:
            min_x = p[0]
        if min_y is None or min_y > p[1]:
            min_y = p[1]
    return min_x, min_y


def print_plot_data(boundary_list):
    boundary_list = boundary_list.copy()
    s = "data:["
    for boundary in boundary_list:
        for i in range(len(boundary)):
            boundary[i] = list(boundary[i])
        s = s + str(boundary) + ","
    s += "],"
    print(s)


def read_train_data(ports_area_input_txt_path, cnt):
    area_list = []
    item_boundary_list_list = []
    names = []
    ports = []
    with open(ports_area_input_txt_path, 'r') as file:
        line_1 = linecache.getline(ports_area_input_txt_path, 1)
        line_1 = line_1.replace('Area:', '').replace(')', ' ').replace(',', '').replace('(', '').replace('\n', '')
        area_point_list = line_1.split(' ')
        area_point_list.pop()
        for i in range(0, len(area_point_list) - 1, 2):
            x = float(area_point_list[i])
            y = float(area_point_list[i + 1])
            area_list.append((x, y))
        txt_data = file.readlines()
        txt_data.pop(0)
        txt_data.pop(0)
        for i in range(0, len(txt_data) - 1, 5):
            item_name = txt_data[i].replace('Module:', '').replace('\n', '')
            item_boundary_list = []
            temp = txt_data[i + 1].replace('Boundary:', '').replace('(', '').replace(')', ' ').replace('\n',
                                                                                                       '').replace(
                ',', '')
            temp = temp.split(' ')
            temp.pop()
            for k in range(0, len(temp) - 1, 2):
                x = float(temp[k])
                y = float(temp[k + 1])
                item_boundary_list.append((x, y))
            item_boundary_list_list.append(item_boundary_list)
            names.append(item_name)
            port_list = [[], [], []]
            for k in range(3):
                temp = txt_data[i + 2 + k].replace('Port:', '').replace('(', '').replace(')', ' ').replace('\n',
                                                                                                           '').replace(
                    ',', '').split(' ')
                temp.pop()
                for m in range(0, len(temp) - 1, 2):
                    temp_list = []
                    x = float(temp[m])
                    y = float(temp[m + 1])
                    temp_list.append(x)
                    temp_list.append(y)
                    port_list[k].append(temp_list)
            ports.append(port_list)
    links = []
    if len(item_boundary_list_list) > 0:
        ports_link_input_txt_path = r'D:\本科\比赛\2022.09.01 EDA图像拼接\2024更新布线算法\data\EDA_DATA\connect\connect_file\connect_' + str(
            cnt) + '.txt'
        with open(ports_link_input_txt_path, 'r') as file:
            file_data = []
            for item in file.readlines():
                item = item.replace('\n', '')
                item = item.replace('\t', ' ')
                item = item.split(' ')
                if item[0][0:4] != 'Link':
                    file_data.append(item)
            for i in range(0, len(file_data) - 1, 2):
                link_dict = {}
                name_list = file_data[i]
                num_list = file_data[i + 1]
                for k in range(len(name_list)):
                    link_dict[name_list[k]] = num_list[k]
                links.append(link_dict)
    return item_boundary_list_list, area_list, links, names, ports


def read_score_data(file_path, cnt):
    line2 = linecache.getline(file_path, 2)
    line3 = linecache.getline(file_path, 3)
    line6 = linecache.getline(file_path, 6)
    line7 = linecache.getline(file_path, 7)
    y = [float(line2.split(':')[1]), float(line3.split(':')[1]), float(line6.split(':')[1]), float(
        line7.split(':')[1])]
    y.append(cnt)
    for data in y:
        if math.isinf(data) or data <= 0:
            return 0
    return calc_score(y, cnt)






def calc_features(item_boundary_list, area_list, links, name_list, ports):
    area_w_h = calc_envelope_rectangle_w_h(area_list)
    area_left_bottom = get_left_bottom_point(area_list)
    S = calc_area(area_w_h[0], area_w_h[1], area_list)
    w_h_list = []
    left_bottom_list = []
    total_s = 0
    total_point = [0, 0]
    cnt = 0
    r_cnt, l_cnt, t_cnt = 0, 0, 0
    min_s = max_s = None
    left_x = right_x = bottom_y = top_y = None
    for i, boundary in enumerate(item_boundary_list):
        for p in boundary:
            if left_x is None or left_x > p[0]:
                left_x = p[0]
            if right_x is None or right_x < p[0]:
                right_x = p[0]
            if bottom_y is None or bottom_y > p[1]:
                bottom_y = p[1]
            if top_y is None or top_y < p[1]:
                top_y = p[1]
        w_h_list.append(calc_envelope_rectangle_w_h(boundary))
        left_bottom_list.append(get_left_bottom_point(boundary))
        s = calc_area(w_h_list[i][0], w_h_list[i][1], boundary)
        if min_s is None or min_s > s:
            min_s = s
        if max_s is None or max_s < s:
            max_s = s
        total_s += s
        if len(boundary) == 4:
            r_cnt += 1
        elif len(boundary) == 6:
            l_cnt += 1
        else:
            t_cnt += 1
        for p in boundary:
            cnt += 1
            total_point = [total_point[0] + p[0], total_point[1] + p[1]]

    rate = total_s / S

    center_point = [total_point[0] / cnt, total_point[1] / cnt]
    crowded_degree = total_s / ((right_x - left_x) * (top_y - bottom_y))
    line_len = 0
    for link in links:
        last_name = None
        for k, v in link.items():
            if last_name is None:
                last_name = k
            else:
                p1 = p2 = None
                for i in range(len(name_list)):
                    if name_list[i] == last_name:
                        p1 = [0, 0]
                        for p in ports[i][int(v) - 1]:
                            p1 = [p1[0] + p[0], p1[1] + p[1]]
                        p1 = [p1[0] / len(ports[i][int(v) - 1]), p1[1] / len(ports[i][int(v) - 1])]
                    elif name_list[i] == k:
                        p2 = [0, 0]
                        for p in ports[i][int(v) - 1]:
                            p2 = [p2[0] + p[0], p2[1] + p[1]]
                        p2 = [p2[0] / len(ports[i][int(v) - 1]), p2[1] / len(ports[i][int(v) - 1])]
                    if p1 is not None and p2 is not None:
                        break
                if p1 is not None and p2 is not None:
                    line_len += (abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]))
    total_spacing = 0
    horizontal_spacing = 0
    vertical_spacing = 0
    for i, boundary_i in enumerate(item_boundary_list):
        lb_point_i = left_bottom_list[i]
        left_space = None
        for j, boundary_j in enumerate(item_boundary_list):
            if i != j:
                lb_point_j = left_bottom_list[j]
                if lb_point_j[0] + w_h_list[0][0] <= lb_point_i[0] and (
                        lb_point_j[1] <= lb_point_i[1] <= lb_point_j[1] + w_h_list[j][1] or lb_point_j[1] <= lb_point_i[
                    1] + w_h_list[i][1] <= lb_point_j[1] + w_h_list[j][1]):
                    if left_space is None or left_space > lb_point_i[0] - (lb_point_j[0] + w_h_list[0][0]):
                        left_space = lb_point_i[0] - (lb_point_j[0] + w_h_list[0][0])
        if left_space is None:
            left_space = lb_point_i[0] - area_left_bottom[0]
        total_spacing += left_space
        horizontal_spacing += left_space
        right_space = None
        for j, boundary_j in enumerate(item_boundary_list):
            if i != j:
                lb_point_j = left_bottom_list[j]
                if lb_point_j[0] >= lb_point_i[0] + w_h_list[i][0] and (
                        lb_point_j[1] <= lb_point_i[1] <= lb_point_j[1] + w_h_list[j][1] or lb_point_j[1] <= lb_point_i[
                    1] + w_h_list[i][1] <= lb_point_j[1] + w_h_list[j][1]):
                    if right_space is None or right_space > lb_point_j[0] - (lb_point_i[0] + w_h_list[i][0]):
                        right_space = lb_point_j[0] - (lb_point_i[0] + w_h_list[i][0])
        if right_space is None:
            right_space = area_left_bottom[0] + area_w_h[0] - lb_point_i[0] - w_h_list[i][0]
        total_spacing += right_space
        horizontal_spacing += right_space
        top_space = None
        for j, boundary_j in enumerate(item_boundary_list):
            if i != j:
                lb_point_j = left_bottom_list[j]
                if lb_point_j[1] >= lb_point_i[1] + w_h_list[i][1] and (
                        lb_point_j[0] <= lb_point_i[0] <= lb_point_j[0] + w_h_list[j][0] or lb_point_j[0] <= lb_point_i[
                    0] + w_h_list[i][0] <= lb_point_j[0] + w_h_list[j][0]):
                    if top_space is None or top_space > lb_point_j[1] - (lb_point_i[1] + w_h_list[i][1]):
                        top_space = lb_point_j[1] - (lb_point_i[1] + w_h_list[i][1])
        if top_space is None:
            top_space = area_w_h[1] + area_left_bottom[1] - lb_point_i[1] - w_h_list[i][1]
        total_spacing += top_space
        vertical_spacing += top_space
        bottom_space = None
        for j, boundary_j in enumerate(item_boundary_list):
            if i != j:
                lb_point_j = left_bottom_list[j]
                if lb_point_j[1] + w_h_list[j][1] <= lb_point_i[1] and (
                        lb_point_j[0] <= lb_point_i[0] <= lb_point_j[0] + w_h_list[j][0] or lb_point_j[0] <= lb_point_i[
                    0] + w_h_list[i][0] <= lb_point_j[0] + w_h_list[j][0]):
                    if bottom_space is None or bottom_space > lb_point_i[1] - (lb_point_j[1] + w_h_list[j][1]):
                        bottom_space = lb_point_i[1] - (lb_point_j[1] + w_h_list[j][1])
        if bottom_space is None:
            bottom_space = lb_point_i[1] - area_left_bottom[1]
        total_spacing += bottom_space
        vertical_spacing += bottom_space

    avg_spacing = total_spacing / len(item_boundary_list) / (area_w_h[0] + area_w_h[1])

    avg_horizontal_spacing = horizontal_spacing / len(item_boundary_list) / area_w_h[0]

    avg_vertical_spacing = vertical_spacing / len(item_boundary_list) / area_w_h[1]

    features = [rate, crowded_degree, line_len, center_point[0], center_point[1], total_s / len(item_boundary_list) / S,
                r_cnt / cnt, l_cnt / cnt, t_cnt / cnt, min_s / max_s, avg_horizontal_spacing, avg_vertical_spacing, cnt]

    return features


def calc_features2(item_boundary_list, area_list, links, name_list, ports):
    features = []
    area_list = correct_boundary(area_list.copy())
    for p in area_list:
        features.append(p[0])
        features.append(p[1])
    for i in range(len(item_boundary_list)):
        item_boundary_list[i], ports[i] = correct_boundary2(item_boundary_list[i].copy(), ports[i].copy())
        for p in item_boundary_list[i]:
            features.append(p[0])
            features.append(p[1])
        for port in ports[i]:
            for p in port:
                features.append(p[0])
                features.append(p[1])
    return features


def calc_features_for_gcn(item_boundary_list, area_list, links, name_list, ports):
    area_w_h = calc_envelope_rectangle_w_h(area_list)
    area_min_x_y = [None, None]
    for p in area_list:
        if area_min_x_y[0] is None or area_min_x_y[0] > p[0]:
            area_min_x_y[0] = p[0]
        if area_min_x_y[1] is None or area_min_x_y[1] > p[1]:
            area_min_x_y[1] = p[1]

    name_index_dict = {}
    for index, name in enumerate(name_list):
        name_index_dict[name] = index

    x = []
    edge_dict = {}
    for i, port in enumerate(ports):
        for j, p in enumerate(port):
            min_x, max_x, min_y, max_y = None, None, None, None
            port_point_arr = []
            for pp in p:
                port_point_arr.append((pp[0] - area_min_x_y[0]) / area_w_h[0])
                port_point_arr.append((pp[1] - area_min_x_y[1]) / area_w_h[1])
                if min_x is None or min_x > pp[0]:
                    min_x = pp[0]
                if max_x is None or max_x < pp[0]:
                    max_x = pp[0]
                if min_y is None or min_y > pp[1]:
                    min_y = pp[1]
                if max_y is None or max_y < pp[1]:
                    max_y = pp[1]

            new_p = [min_x + (max_x - min_x) / 2, min_y + (max_y - min_y)]
            w, h = (max_x - min_x), (max_y - min_y)
            new_p = [new_p[0] - area_min_x_y[0], new_p[1] - area_min_x_y[1]]
            new_p = [new_p[0] / area_w_h[0], new_p[1] / area_w_h[1], len(p) / 8.0,
                     calc_area(w, h, p) / (area_w_h[0] * area_w_h[1])]

            l = len(new_p)

            new_p.extend(port_point_arr)

            while len(new_p) < l + 3 * 8:
                new_p.append(0)

            if len(new_p) > l + 3 * 8:
                raise RuntimeError

            for n in new_p:
                if n < 0:
                    return []
            edge_dict[name_list[i] + "-" + str(j)] = len(x)
            x.append(new_p)
    edge_index = [[], []]
    for link in links:
        last_point = None
        for key, value in link.items():
            if last_point is None:
                last_point = [key, value]
            else:
                try:
                    edge_index[1].append(edge_dict[key + "-" + str(int(value) - 1)])
                    edge_index[0].append(edge_dict[last_point[0] + "-" + str(int(last_point[1]) - 1)])
                    last_point = [key, value]
                except:
                    last_point = None
                    if len(edge_index[1]) > len(edge_index[0]):
                        edge_index[1].pop(len(edge_index[1]) - 1)
                    elif len(edge_index[1]) < len(edge_index[0]):
                        edge_index[0].pop(len(edge_index[0]) - 1)


    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return [x, edge_index]


def correct_boundary(boundary):
    min_x = min_y = None
    for p in boundary:
        if min_x is None or min_x > p[0]:
            min_x = p[0]
        if min_y is None or min_y > p[1]:
            min_y = p[1]
    return [[p[0] - min_x, p[1] - min_y] for p in boundary]


def correct_boundary2(boundary, ports):
    min_x = min_y = None
    for p in boundary:
        if min_x is None or min_x > p[0]:
            min_x = p[0]
        if min_y is None or min_y > p[1]:
            min_y = p[1]
    for i in range(len(ports)):
        for j in range(len(ports[i])):
            ports[i][j] = [ports[i][j][0] - min_x, ports[i][j][1] - min_y]
    return [[p[0] - min_x, p[1] - min_y] for p in boundary], ports


def calc_score(predict, l):
    predict_score = 0
    alpha = 10000.0
    if l < 25:
        if (1 - predict[0]) * l <= 2:
            s1 = 50
            predict_score += s1
            s2 = 2 * alpha / predict[3]
            predict_score += s2
            s3 = alpha / predict[2]
            predict_score += s3
        else:
            s1 = 50 * predict[0] * predict[1]
            predict_score += s1
    elif 25 <= l <= 50:
        if predict[0] >= 0.9:
            s1 = 70
            predict_score += s1
            s2 = 2 * alpha / predict[3]
            predict_score += s2
            s3 = alpha / predict[2]
            predict_score += s3
        else:
            s1 = 70 * predict[0] * predict[1]
            predict_score += s1
    else:
        if predict[0] >= 0.85:
            s1 = 90
            predict_score += s1
            s2 = 2 * alpha / predict[3]
            predict_score += s2
            s3 = alpha / predict[2]
            predict_score += s3
        else:
            s1 = 90 * predict[0] * predict[1]
            predict_score += s1
    if predict_score == float('inf'):
        print(predict)
    return predict_score


def calc_center_position(points):
    center = [0, 0]
    for p in points:
        center = [center[0] + p[0], center[1] + p[1]]
    return [center[0] / len(points), center[1] / len(points)]


class MyDataSet(Dataset):
    def __init__(self, x):
        self.data = x

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def min_max_scaler(num, min_m, max_m):
    if num > max_m or num < min_m:
        raise RuntimeError("超出区间,num:", num, "min:", min_m, "max:", max_m)
    return (num - min_m) / (max_m - min_m)


def get_features_for_ml_gcn(gcn_model, x_gcn_data, x_ml_data):
    if len(x_ml_data) == 0:
        return []
    x_data = [Data(x=x_gcn_data[0], edge_index=x_gcn_data[1])]
    data_set = MyDataSet(x_data)
    data_loader = DataLoader(data_set, batch_size=1, shuffle=True)
    for data in data_loader:
        x = torch.relu(gcn_model.conv1(data.x, data.edge_index))
        x = gcn_model.conv2(x, data.edge_index)
        x = global_mean_pool(x, data.batch)
        x = x.squeeze(0).tolist()
        x.extend(x_ml_data)
        return x