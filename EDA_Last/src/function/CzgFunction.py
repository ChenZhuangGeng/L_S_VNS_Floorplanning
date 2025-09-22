# -*-coding:utf-8-*-
# Author: WSKH
# Blog: wskh0929.blog.csdn.net
# Time: 2022/9/1 12:07
import sys
sys.path.append('/home/eda220806/EDA_Last')
from src.classes.EdaClasses import *
import random, itertools, linecache


# 读取ICEEC2020_5.txt数据
def read_items_from_ICEEC(path):
    return_data = []
    file = open(path, 'r')
    file_data = file.readlines()
    i = 0
    count = 0
    for row in file_data:
        boundary = []
        i = i + 1
        if i % 2 == 1:
            continue
        # print('row',i)
        row = row.replace('(', '')
        row = row.replace(')', ',')
        row = row.replace('\n', '')
        locations = row.split(',')
        del (locations[-1])
        # print(locations)
        m = 0
        x_list = []
        y_list = []
        while m < len(locations):
            x = int(locations[m])
            x_list.append(x)
            y = int(locations[m + 1])
            y_list.append(y)
            m += 2
            location = (x, y)
            boundary.append(location)
        min_x = min(x_list)
        min_y = min(y_list)
        boundary = [(x - min_x, y - min_y) for x, y in boundary]

        # print(boundary)
        item = Item('TEST' + str(count), boundary, [])
        count += 1
        item.init_item()
        return_data.append(item)
    # print(return_data[0].to_string())
    return return_data


# 根据旋转类型，返回旋转后的模块
def calc_rotate_item(w, h, boundary, ports, orient):
    if orient.value == "R90":
        boundary = coordinate_transformation(boundary, h)
        ports = coordinate_transformation(ports, h)
        temp = h
        h = w
        w = temp
    elif orient.value == "R180":
        for i in range(2):
            boundary = coordinate_transformation(boundary, h)
            ports = coordinate_transformation(ports, h)
            temp = h
            h = w
            w = temp
    elif orient.value == "R270":
        for i in range(3):
            boundary = coordinate_transformation(boundary, h)
            ports = coordinate_transformation(ports, h)
            temp = h
            h = w
            w = temp
    elif orient.value == "MX":
        boundary = mirror_flip_X(boundary, h)
        ports = mirror_flip_X(ports, h)
    elif orient.value == "MY":
        boundary = mirror_flip_Y(boundary, w)
        ports = mirror_flip_Y(ports, w)
    elif orient.value == "MXR90":
        boundary = mirror_flip_X(boundary, h)
        ports = mirror_flip_X(ports, h)
        boundary = coordinate_transformation(boundary, h)
        ports = coordinate_transformation(ports, h)
        temp = h
        h = w
        w = temp
    elif orient.value == "MYR90":
        boundary = mirror_flip_Y(boundary, w)
        ports = mirror_flip_Y(ports, w)
        boundary = coordinate_transformation(boundary, h)
        ports = coordinate_transformation(ports, h)
        temp = h
        h = w
        w = temp
    return w, h, boundary, orient, ports


# X轴镜像翻转
def mirror_flip_X(boundary, h):
    new_boundary_list = []
    # 对每个坐标依次进行变化
    for location in boundary:
        if type(location) is list:
            lst = []
            for loc in location:
                new_x = loc[0]
                new_y = h - loc[1]
                new_location = (new_x, new_y)
                lst.append(new_location)
            new_boundary_list.append(lst)
        else:
            new_x = location[0]
            new_y = h - location[1]
            new_location = (new_x, new_y)
            new_boundary_list.append(new_location)
    return new_boundary_list


# Y轴镜像翻转
def mirror_flip_Y(boundary, w):
    boundary_list = boundary
    new_boundary_list = []
    # 对每个坐标依次进行变化
    for location in boundary_list:
        if type(location) is list:
            lst = []
            for loc in location:
                new_x = w - loc[0]
                new_y = loc[1]
                new_location = (new_x, new_y)
                lst.append(new_location)
            new_boundary_list.append(lst)
        else:
            new_x = w - location[0]
            new_y = location[1]
            new_location = (new_x, new_y)
            new_boundary_list.append(new_location)
    return new_boundary_list


# 逆时针90度坐标变换(边界，包络矩形的高)
def coordinate_transformation(boundary, h):
    new_boundary_list = []
    # 对每个坐标依次进行变化
    for location in boundary:
        if type(location) is list:
            lst = []
            for loc in location:
                old_x = loc[0]
                old_y = loc[1]
                new_x = h - old_y
                new_y = old_x
                new_location = (new_x, new_y)
                lst.append(new_location)
            new_boundary_list.append(lst)
        else:
            old_x = location[0]
            old_y = location[1]
            new_x = h - old_y
            new_y = old_x
            new_location = (new_x, new_y)
            new_boundary_list.append(new_location)
    # # new_boundary_list的第一个要调到最后去,不用了能连起来就行
    # k = new_boundary_list.pop(0)
    # new_boundary_list.append(k)
    return new_boundary_list


# 读取数据，返回Instance对象
def get_instance(ports_area_input_txt_path, ports_link_input_txt_path):
    link_list = read_data_port_link(ports_link_input_txt_path)
    item_list, rule_dict, area_list = read_data_port_area(ports_area_input_txt_path)
    instance = Instance(item_list, link_list, rule_dict, area_list)
    # print('items', instance.items[0].boundary)
    # print('links', instance.links[0].link_dict)
    # print('rule', instance.rule)
    # print('area', instance.area)
    return instance


# 读取模块文件格式
def read_data_port_area(ports_area_input_txt_path):
    return_data = []
    area_list = []
    rule_dict = {}
    item_list = []
    ### 读取内容
    file = open(ports_area_input_txt_path, 'r')
    ## 先读前两行
    # 第一行
    line_1 = linecache.getline(ports_area_input_txt_path, 1)
    line_1 = line_1.replace('Area:', '').replace(')', ' ').replace(',', '').replace('(', '').replace('\n', '')
    area_point_list = line_1.split(' ')
    area_point_list.pop()
    for i in range(0, len(area_point_list) - 1, 2):
        x = float(area_point_list[i])
        y = float(area_point_list[i + 1])
        area_list.append((x, y))
    # 第二行
    line_2 = linecache.getline(ports_area_input_txt_path, 2)
    line_2 = line_2.replace('Rule:', '').replace(';', ' ').replace('(', ' (').replace('\n', '')
    rule = line_2.split(' ')
    for i in range(0, len(rule) - 1, 2):
        key = rule[i]
        value = rule[i + 1]
        if ',' in value:
            value = value.replace('(', '').replace(',', ' ').replace(')', '')
            temp = value.split(' ')
            x = int(temp[0])
            y = int(temp[1])
            rule_dict[key] = (x, y)
        else:
            rule_dict[key] = float(value[1:4])
    ## 模块
    txt_data = file.readlines()
    txt_data.pop(0)
    txt_data.pop(0)

    for i in range(0, len(txt_data) - 1, 5):
        # print(i, txt_data[i])
        # print(i+1, txt_data[i+1])
        # print(i+2, txt_data[i+2])
        # print(i+3, txt_data[i+3])
        # print(i+4, txt_data[i+4])
        # print('==================')
        # name
        item_name = txt_data[i].replace('Module:', '').replace('\n', '')
        # boundary
        item_boundary_list = []
        temp = txt_data[i + 1].replace('Boundary:', '').replace('(', '').replace(')', ' ').replace('\n', '').replace(
            ',', '')
        temp = temp.split(' ')
        temp.pop()
        x_list = []
        y_list = []
        for k in range(0, len(temp) - 1, 2):
            x = float(temp[k])
            x_list.append(x)
            y = float(temp[k + 1])
            y_list.append(y)
            item_boundary_list.append((x, y))
        x_min = min(x_list)
        y_min = min(y_list)
        item_boundary_list = [(x - x_min, y - y_min) for x, y in item_boundary_list]
        # port 貌似都是3个，这是固定的
        port_list = [[], [], []]
        for k in range(3):
            temp = txt_data[i + 2 + k].replace('Port:', '').replace('(', '').replace(')', ' ').replace('\n',
                                                                                                       '').replace(',',
                                                                                                                   '').split(
                ' ')
            temp.pop()
            for m in range(0, len(temp) - 1, 2):
                x = float(temp[m])
                y = float(temp[m + 1])
                port_list[k].append((x, y))
            port_list[k] = [(x - x_min, y - y_min) for x, y in port_list[k]]
        # item
        # print('item_name == ', item_name)
        # print('item_boundary_list == ', item_boundary_list)
        # print('port_list == ', port_list)
        # print('==========================================')
        item = Item(item_name, item_boundary_list, port_list)
        item.init_item()
        item_list.append(item)
    # print(len(item_list))
    return_data.append(item_list)
    return_data.append(rule_dict)
    return_data.append(area_list)
    return return_data


# 读取连接文件格式
def read_data_port_link(ports_link_input_txt_path):
    return_data = []
    # 读取内容
    file = open(ports_link_input_txt_path, 'r')
    file_data = []
    for item in file.readlines():
        item = item.replace('\n', '')
        item = item.replace('\t', ' ')
        item = item.split(' ')
        if item[0][0:4] != 'Link':
            file_data.append(item)
    # 生成对象
    for i in range(0, len(file_data) - 1, 2):
        link_dict = {}
        # file_data[i].pop()
        # file_data[i+1].pop()
        name_list = file_data[i]
        num_list = file_data[i + 1]
        for k in range(len(name_list)):
            link_dict[name_list[k]] = num_list[k]
        link = Link(link_dict)
        return_data.append(link)
    file.close()
    return return_data


# 随机生成数据，包括模块文件和连接文件
def generate_data():
    module_count = 7
    port_count = 3
    port_size = 5
    link_count = 3
    write_data_port_area(module_count, port_count, port_size)
    write_data_port_link(module_count, link_count, port_count)
    return


# 生成模块文件格式
def write_data_port_area(module_count, port_count, port_size):
    note = open('Ports_area_etc_input_1.txt', mode='w', encoding='utf8')
    ## 大矩形,需要设定大矩形范围
    big_rectangular_size = random.randint(500, 1000)
    note.writelines(
        ['Area:', '(0, 0)', '(', str(big_rectangular_size), ',', str(0), ')', '(', str(big_rectangular_size), ',',
         str(big_rectangular_size),
         ')', '(', str(0), ',', str(big_rectangular_size), ')', '\n'])
    note.writelines(
        ['Rule:', 'SD', '(5, 5)', ';', 'GATE', '(4, 4)', ';', 'SD_GATE', '(', '0.5', ')', ';', 'SD_ITO', '(', '0.5',
         ')',
         ';',
         'GATE_ITO', '(', '0.5', ')', '\n'])

    ## 模块，暂时固定3个
    # module_count = 5
    # port_size = 5
    # port_count = 3
    for i in range(module_count):
        note.writelines(['Module:', 'M', str(i + 1), '\n'])
        w = 10000
        h = 0
        r1 = 0
        r2 = 0
        r3 = 0
        r4 = 0
        r5 = 0
        r6 = 0
        while w < r3 + port_size or h <= port_size or r2 < port_size or h - port_size < r2:
            # 边界,暂定L型
            h = random.randint(int(big_rectangular_size / 10), int(big_rectangular_size / 5))
            w = random.randint(int(big_rectangular_size / 10), int(big_rectangular_size / 5))
            r1 = w
            r2 = random.randint(1, int(big_rectangular_size / 10))
            while r2 >= h:
                r2 = random.randint(1, int(big_rectangular_size / 10))
            r3 = random.randint(1, int(big_rectangular_size / 10))
            while r3 > w or r3 < port_size:
                r3 = random.randint(1, int(big_rectangular_size / 10))
            r4 = h - r2
            r5 = w - r3
            r6 = h
        note.writelines(
            ['Boundary:', '(0, 0)(', str(r1), ',0)(', str(r1), ',', str(r2), ')(', str(r5), ',', str(r2), ')(', str(r5),
             ',',
             str(r6), ')(0,',
             str(r6), ');GATE', '\n'])
        # 端口，暂定3个,接口宽高暂定5
        for k in range(port_count):
            which_one = random.randint(1, 6)
            if which_one == 1:
                x = 0
                y = random.randint(0, h - port_size)
            elif which_one == 2:
                x = random.randint(0, w - r3 - port_size)
                y = h - port_size
            elif which_one == 3:
                x = w - r3 - port_size
                y = random.randint(r2, h - port_size)
            elif which_one == 4:
                x = random.randint(w - r3, w - port_size)
                y = r2 - port_size
            elif which_one == 5:
                x = w - port_size
                y = random.randint(0, r2 - port_size)
            elif which_one == 6:
                x = random.randint(0, w - port_size)
                y = 0
            note.writelines(
                ['Port:(', str(x), ',', str(y), ')(', str(x + port_size), ',', str(y), ')(', str(x + port_size), ',',
                 str(y + port_size), ')(', str(x), ',', str(y + port_size), ');SD', '\n'])

    note.close()


# 生成连接文件格式
def write_data_port_link(module_count, link_count, port_count):
    set_list = subsets(range(module_count))
    # 初始化set_A,允许使用的子集字典,set_write
    set_dict = {}
    index = 0
    for item in set_list:
        set_dict[item] = 1
    set_A = set_list[random.randint(0, len(set_list) - 1)]
    set_write = [set_A]
    # 更新A允许的子集字典
    for item in set_list:
        if extra_same_elem(subsets(item), subsets(set_A)):
            set_dict[item] = 0
    # 循环生成新子集
    for i in range(link_count - 1):
        set_B = set_list[random.randint(0, len(set_list) - 1)]
        # 子集列表set_list中随机抽一个B，如果不符合以下要求就重新抽，set_dict[set_B] == 0:是为了不与前面的使用过的子集冲突
        while extra_same_elem(subsets(set_B), subsets(set_A)) or extra_same_elem(subsets(set_B),
                                                                                 subsets(set_write[index])) or set_dict[
            set_B] == 0:
            set_B = set_list[random.randint(0, len(set_list) - 1)]
        set_write.append(set_B)
        # 更新B允许的子集字典
        for item in set_list:
            if extra_same_elem(subsets(item), subsets(set_B)):
                set_dict[item] = 0
        index += 1  # 索引加一
    # 书写txt
    note = open('Ports_link_input_1.txt', mode='w', encoding='utf8')
    for i in range(link_count):
        note.writelines(['Link', str(i + 1), ':\n'])
        for index in set_write[i]:
            note.writelines(['M', str(index), '\t'])
        note.writelines(['\n'])
        for index in set_write[i]:
            port_num = random.randint(1, port_count)
            note.writelines(str(port_num) + '\t')
        note.writelines(['\n'])
    note.close()
    return


# 返回两个单位以上的集合
def subsets(nums):
    if len(nums) == 2:
        return tuple(nums)
    res = []
    for i in range(len(nums) + 1):
        for tmp in itertools.combinations(nums, i):
            res.append(tmp)
    index_list = []
    for i in range(len(res) - 1):
        if len(res[i]) != 0 and len(res[i]) != 1:
            index_list.append(i)
    new_res = []
    for i in index_list:
        new_res.append(res[i])
    return new_res


# 对比两个列表中是否有重复元素,用来对比各自的子集是否相同
def extra_same_elem(list1, list2):
    if type(list1) == tuple:
        set1 = {list1}
    else:
        set1 = set(list1)
    if type(list2) == tuple:
        set2 = {list2}
    else:
        set2 = set(list2)
    iset = set1.intersection(set2)
    flag = False
    if len(iset) > 0:
        flag = True
    return flag


# 传入两个rotate_item对象，判断他们是否重叠
def is_overlap(rotate_item_1, left_bottom_point_1, rotate_item_2, left_bottom_point_2):
    # left_x_1 = left_bottom_point_1[0]
    # left_y_1 = left_bottom_point_1[1]
    # left_x_2 = left_bottom_point_2[0]
    # left_y_2 = left_bottom_point_2[1]
    # # L和L
    # if len(rotate_item_1.boundary) == 6 and len(rotate_item_2.boundary) == 6:
    #     L_dict_1 = get_peak_gap_point(rotate_item_1, left_x_1, left_y_1)
    #     L_dict_2 = get_peak_gap_point(rotate_item_2, left_x_2, left_y_2)
    #     if is_in_gap(L_dict_1['peak_point'], L_dict_2['gap_point_1'], L_dict_2['gap_point_2'], L_dict_2['gap_point_3']):
    #         # 在缺口中，没重叠
    #         return False
    #     elif is_in_gap(L_dict_2['peak_point'], L_dict_1['gap_point_1'], L_dict_1['gap_point_2'],
    #                    L_dict_1['gap_point_3']):
    #         # 在缺口中，没重叠
    #         return False
    #     else:
    #         return True
    # # L和T
    # elif len(rotate_item_1.boundary) == 6 and len(rotate_item_2.boundary) == 8:
    #     L_dict = get_peak_gap_point(rotate_item_1, left_x_1, left_y_1)
    #     T_dict = get_peak_gap_point(rotate_item_2, left_x_2, left_y_2)
    #     if is_in_gap(T_dict['peak_point_1'], L_dict['gap_point_1'], L_dict['gap_point_2'],
    #                  L_dict['gap_point_3']) or is_in_gap(T_dict['peak_point_2'], L_dict['gap_point_1'],
    #                                                      L_dict['gap_point_2'], L_dict['gap_point_3']):
    #         # 在缺口中
    #
    #         return False
    #
    #     elif is_in_gap(L_dict['peak_point'], T_dict['gap_point_1'], T_dict['gap_point_2'],
    #                    T_dict['gap_point_3'] or
    #                    L_dict['peak_point'], T_dict['gap_point_4'], T_dict['gap_point_5'],
    #                    T_dict['gap_point_6']):
    #         return False
    #     else:
    #         return True
    # # T和L
    # elif len(rotate_item_1.boundary) == 8 and len(rotate_item_2.boundary) == 6:
    #     T_dict = get_peak_gap_point(rotate_item_1, left_x_1, left_y_1)
    #     L_dict = get_peak_gap_point(rotate_item_2, left_x_2, left_y_2)
    #     if is_in_gap(T_dict['peak_point_1'], L_dict['gap_point_1'], L_dict['gap_point_2'],
    #                  L_dict['gap_point_3']) or is_in_gap(T_dict['peak_point_2'], L_dict['gap_point_1'],
    #                                                      L_dict['gap_point_2'], L_dict['gap_point_3']):
    #         # 在缺口中
    #         return False
    #     elif is_in_gap(L_dict['peak_point'], T_dict['gap_point_1'], T_dict['gap_point_2'],
    #                    T_dict['gap_point_3'] or
    #                    L_dict['peak_point'], T_dict['gap_point_4'], T_dict['gap_point_5'],
    #                    T_dict['gap_point_6']):
    #         return False
    #     else:
    #         return True
    # # T和T
    # elif len(rotate_item_1.boundary) == 8 and len(rotate_item_2.boundary) == 8:
    #     T_dict_1 = get_peak_gap_point(rotate_item_1, left_x_1, left_y_1)
    #     T_dict_2 = get_peak_gap_point(rotate_item_2, left_x_2, left_y_2)
    #     # 1 in 2
    #     if is_in_gap(T_dict_1['peak_point_1'], T_dict_2['gap_point_1'], T_dict_2['gap_point_2'],
    #                  T_dict_2['gap_point_3']) or is_in_gap(T_dict_1['peak_point_1'], T_dict_2['gap_point_4'],
    #                                                        T_dict_2['gap_point_5'],
    #                                                        T_dict_2['gap_point_6']) or is_in_gap(
    #         T_dict_1['peak_point_2'], T_dict_2['gap_point_1'], T_dict_2['gap_point_2'],
    #         T_dict_2['gap_point_3']) or is_in_gap(T_dict_1['peak_point_2'], T_dict_2['gap_point_4'],
    #                                               T_dict_2['gap_point_5'],
    #                                               T_dict_2['gap_point_6']):
    #         return False
    #     # 2 in 1
    #     elif is_in_gap(T_dict_2['peak_point_1'], T_dict_1['gap_point_1'], T_dict_1['gap_point_2'],
    #                    T_dict_1['gap_point_3']) or is_in_gap(T_dict_2['peak_point_1'], T_dict_1['gap_point_4'],
    #                                                          T_dict_1['gap_point_5'],
    #                                                          T_dict_1['gap_point_6']) or is_in_gap(
    #         T_dict_2['peak_point_2'], T_dict_1['gap_point_1'], T_dict_1['gap_point_2'],
    #         T_dict_1['gap_point_3']) or is_in_gap(T_dict_2['peak_point_2'], T_dict_1['gap_point_4'],
    #                                               T_dict_1['gap_point_5'],
    #                                               T_dict_1['gap_point_6']):
    #         return False
    #     else:
    #         return True
    # # L和R
    # elif len(rotate_item_1.boundary) == 6 and len(rotate_item_2.boundary) == 4:
    #     L_dict = get_peak_gap_point(rotate_item_1, left_x_1, left_y_1)
    #     R_dict = get_peak_gap_point(rotate_item_2, left_x_2, left_y_2)
    #     # L大R小,R四个顶点都要试
    #     # todo 不能单靠一个点在缺口就判定不重叠！！！
    #     if is_in_gap(R_dict['peak_point_1'], L_dict['gap_point_1'], L_dict['gap_point_2'],
    #                  L_dict['gap_point_3']) or is_in_gap(R_dict['peak_point_2'], L_dict['gap_point_1'],
    #                                                      L_dict['gap_point_2'], L_dict['gap_point_3']) or is_in_gap(
    #         R_dict['peak_point_3'], L_dict['gap_point_1'], L_dict['gap_point_2'], L_dict['gap_point_3']) or is_in_gap(
    #         R_dict['peak_point_4'], L_dict['gap_point_1'], L_dict['gap_point_2'], L_dict['gap_point_3']):
    #         return False
    #     else:
    #         return True
    # # R和L
    # elif len(rotate_item_1.boundary) == 4 and len(rotate_item_2.boundary) == 8:
    #     R_dict = get_peak_gap_point(rotate_item_1, left_x_1, left_y_1)
    #     L_dict = get_peak_gap_point(rotate_item_2, left_x_2, left_y_2)

    # if len(rotate_item_2.boundary) == 4 or len(rotate_item_1.boundary) == 4:
    #     return False
    peak_point_list_1 = get_peak_point_list(rotate_item_1, left_bottom_point_1)
    # peak_point_list_2 = get_peak_point_list(rotate_item_2, left_bottom_point_2)
    # envelope_point_list_1 = get_envelope_point(rotate_item_1, left_bottom_point_1)
    envelope_point_list_2 = get_envelope_point(rotate_item_2, left_bottom_point_2)
    in_envelope_area_peak_point_1in2 = []
    # in_envelope_area_peak_point_2in1 = []
    if len(rotate_item_1.boundary) == 6:
        gap_point_list_1 = get_gap_point_L(rotate_item_1, left_bottom_point_1)
    elif len(rotate_item_1.boundary) == 8:
        gap_point_list_1 = get_gap_point_T(rotate_item_1, left_bottom_point_1)
    if len(rotate_item_2.boundary) == 6:
        gap_point_list_2 = get_gap_point_L(rotate_item_2, left_bottom_point_2)
    elif len(rotate_item_2.boundary) == 8:
        gap_point_list_2 = get_gap_point_T(rotate_item_2, left_bottom_point_2)
    count_1in2L = 0
    count_1in2_T1 = 0
    count_1in2_T2 = 0
    ## 1在2中
    for peak_point in peak_point_list_1:
        if is_in_envelope_rectangle(peak_point, envelope_point_list_2[0], envelope_point_list_2[1],
                                    envelope_point_list_2[2], envelope_point_list_2[3]):
            in_envelope_area_peak_point_1in2.append(peak_point)
    # 2是L
    if len(rotate_item_2.boundary) == 6:
        for envelope_area_peak_point in in_envelope_area_peak_point_1in2:
            if is_in_envelope_rectangle(envelope_area_peak_point, gap_point_list_2[0], gap_point_list_2[1],
                                        gap_point_list_2[2], gap_point_list_2[3]):
                count_1in2L += 1
        if count_1in2L != len(in_envelope_area_peak_point_1in2):
            # 重叠
            return True
    # 2是T
    elif len(rotate_item_2.boundary) == 8:
        for envelope_area_peak_point in in_envelope_area_peak_point_1in2:
            # 2的第一个缺口
            if is_in_envelope_rectangle(envelope_area_peak_point, gap_point_list_2[0][0], gap_point_list_2[0][1],
                                        gap_point_list_2[0][2], gap_point_list_2[0][3]):
                count_1in2_T1 += 1
            if is_in_envelope_rectangle(envelope_area_peak_point, gap_point_list_2[1][0], gap_point_list_2[1][1],
                                        gap_point_list_2[1][2], gap_point_list_2[1][3]):
                count_1in2_T2 += 1
            if count_1in2_T1 > 0 and count_1in2_T2 == 0:
                if count_1in2_T1 != len(in_envelope_area_peak_point_1in2):
                    return True
            elif count_1in2_T1 == 0 and count_1in2_T2 > 0:
                if count_1in2_T2 != len(in_envelope_area_peak_point_1in2):
                    return True
            elif count_1in2_T1 == 0 and count_1in2_T2 == 0:
                if len(in_envelope_area_peak_point_1in2) > 0:
                    return True
            elif count_1in2_T1 > 0 and count_1in2_T2 > 0:
                return True
    # 2是矩形
    elif len(rotate_item_2.boundary) == 4:
        return '2是矩形，不要传进来！！'
    else:
        return False


# 返回绝对位置的顶点
def get_peak_point_list(rotate_item_1, left_bottom_point_1):
    peak_point_list = []
    for point in rotate_item_1.boundary:
        x = point[0] + left_bottom_point_1[0]
        y = point[1] + left_bottom_point_1[0]
        peak_point_list.append((x, y))
    return peak_point_list


# 返回绝对位置的包络矩形点
def get_envelope_point(rotate_item_1, left_bottom_point_1):
    envelope_point_list = []
    x = left_bottom_point_1[0]
    y = left_bottom_point_1[0]
    if len(rotate_item_1.boundary) == 4:
        envelope_point_list = get_peak_point_list(rotate_item_1, left_bottom_point_1)
    elif len(rotate_item_1.boundary) == 6 or len(rotate_item_1.boundary) == 8:
        envelope_point_list.append((x, y))
        envelope_point_list.append((x + rotate_item_1.w, y))
        envelope_point_list.append((x + rotate_item_1.w, y + rotate_item_1.h))
        envelope_point_list.append((x, y + rotate_item_1.h))
    return envelope_point_list


def get_gap_point_L(rotate_item_1, left_bottom_point_1):
    L_return_list = []
    left_x_1 = left_bottom_point_1[0]
    left_y_1 = left_bottom_point_1[1]
    # 1是L型
    if len(rotate_item_1.boundary) == 6:
        # 0度
        if rotate_item_1.type == '0':
            # 缺口四点
            gap_point_1_x = left_x_1 + rotate_item_1.w1
            gap_point_1_y = left_y_1 + rotate_item_1.h
            gap_point_2_x = left_x_1 + rotate_item_1.w1
            gap_point_2_y = left_y_1 + rotate_item_1.h2
            gap_point_3_x = left_x_1 + rotate_item_1.w
            gap_point_3_y = left_y_1 + rotate_item_1.h2
            gap_point_4_x = left_x_1 + rotate_item_1.w
            gap_point_4_y = left_y_1 + rotate_item_1.h
        # 90度
        elif rotate_item_1.type == '90':
            gap_point_1_x = left_x_1 + rotate_item_1.w
            gap_point_1_y = left_y_1 + rotate_item_1.h1
            gap_point_2_x = left_x_1 + rotate_item_1.w1
            gap_point_2_y = left_y_1 + rotate_item_1.h1
            gap_point_3_x = left_x_1 + rotate_item_1.w1
            gap_point_3_y = left_y_1
            gap_point_4_x = left_x_1 + rotate_item_1.w
            gap_point_4_y = left_y_1
        # 180度
        elif rotate_item_1.type == '180':
            gap_point_1_x = left_x_1
            gap_point_1_y = left_y_1 + rotate_item_1.h2
            gap_point_2_x = left_x_1 + rotate_item_1.w1
            gap_point_2_y = left_y_1 + rotate_item_1.h2
            gap_point_3_x = left_x_1 + rotate_item_1.w1
            gap_point_3_y = left_y_1
            gap_point_4_x = left_x_1
            gap_point_4_y = left_y_1
        # 270度
        elif rotate_item_1.type == '270':
            gap_point_1_x = left_x_1
            gap_point_1_y = left_y_1 + rotate_item_1.h1
            gap_point_2_x = left_x_1 + rotate_item_1.w1
            gap_point_2_y = left_y_1 + rotate_item_1.h1
            gap_point_3_x = left_x_1 + rotate_item_1.w1
            gap_point_3_y = left_y_1 + rotate_item_1.h
            gap_point_4_x = left_x_1
            gap_point_4_y = left_y_1 + rotate_item_1.h
        L_return_list.append((gap_point_1_x, gap_point_1_y))
        L_return_list.append((gap_point_2_x, gap_point_2_y))
        L_return_list.append((gap_point_3_x, gap_point_3_y))
        L_return_list.append((gap_point_4_x, gap_point_4_y))
        return L_return_list
    else:
        print('传进来的不是L')
        return False


def get_gap_point_T(rotate_item_1, left_bottom_point_1):
    T_return_list = [[], []]
    left_x_1 = left_bottom_point_1[0]
    left_y_1 = left_bottom_point_1[1]
    if len(rotate_item_1.boundary) == 8:
        # 0度
        if rotate_item_1.type == '0':
            # 缺口八点
            gap_point_1_x = left_x_1
            gap_point_1_y = left_y_1 + rotate_item_1.h1
            gap_point_2_x = left_x_1 + rotate_item_1.w1
            gap_point_2_y = left_y_1 + rotate_item_1.h1
            gap_point_3_x = left_x_1 + rotate_item_1.w1
            gap_point_3_y = left_y_1 + rotate_item_1.h
            gap_point_4_x = left_x_1
            gap_point_4_y = left_y_1 + rotate_item_1.h

            gap_point_5_x = left_x_1 + rotate_item_1.w1 + rotate_item_1.w2
            gap_point_5_y = left_y_1 + rotate_item_1.h
            gap_point_6_x = left_x_1 + rotate_item_1.w1 + rotate_item_1.w2
            gap_point_6_y = left_y_1 + rotate_item_1.h4
            gap_point_7_x = left_x_1 + rotate_item_1.w
            gap_point_7_y = left_y_1 + rotate_item_1.h4
            gap_point_8_x = left_x_1 + rotate_item_1.w
            gap_point_8_y = left_y_1 + rotate_item_1.h
        # 90度
        elif rotate_item_1.type == '90':
            gap_point_1_x = left_x_1 + rotate_item_1.w4
            gap_point_1_y = left_y_1 + rotate_item_1.h
            gap_point_2_x = left_x_1 + rotate_item_1.w4
            gap_point_2_y = left_y_1 + rotate_item_1.h2 + rotate_item_1.h3
            gap_point_3_x = left_x_1 + rotate_item_1.w
            gap_point_3_y = left_y_1 + rotate_item_1.h2 + rotate_item_1.h3
            gap_point_4_x = left_x_1 + rotate_item_1.w
            gap_point_4_y = left_y_1 + rotate_item_1.h

            gap_point_5_x = left_x_1 + rotate_item_1.w
            gap_point_5_y = left_y_1 + rotate_item_1.h3
            gap_point_6_x = left_x_1 + rotate_item_1.w1
            gap_point_6_y = left_y_1 + rotate_item_1.h3
            gap_point_7_x = left_x_1 + rotate_item_1.w1
            gap_point_7_y = left_y_1
            gap_point_8_x = left_x_1 + rotate_item_1.w
            gap_point_8_y = left_y_1
        # 180度
        elif rotate_item_1.type == '180':
            gap_point_1_x = left_x_1
            gap_point_1_y = left_y_1 + rotate_item_1.h2
            gap_point_2_x = left_x_1 + rotate_item_1.w1
            gap_point_2_y = left_y_1 + rotate_item_1.h2
            gap_point_3_x = left_x_1 + rotate_item_1.w1
            gap_point_3_y = left_y_1
            gap_point_4_x = left_x_1
            gap_point_4_y = left_y_1

            gap_point_5_x = left_x_1 + rotate_item_1.w
            gap_point_5_y = left_y_1 + rotate_item_1.h3
            gap_point_6_x = left_x_1 + rotate_item_1.w1 + rotate_item_1.w2
            gap_point_6_y = left_y_1 + rotate_item_1.h3
            gap_point_7_x = left_x_1 + rotate_item_1.w1 + rotate_item_1.w2
            gap_point_7_y = left_y_1
            gap_point_8_x = left_x_1 + rotate_item_1.w
            gap_point_8_y = left_y_1
        # 270度
        elif rotate_item_1.type == '270':
            gap_point_1_x = left_x_1 + rotate_item_1.w3
            gap_point_1_y = left_y_1 + rotate_item_1.h
            gap_point_2_x = left_x_1 + rotate_item_1.w3
            gap_point_2_y = left_y_1 + rotate_item_1.h1 + rotate_item_1.h2
            gap_point_3_x = left_x_1
            gap_point_3_y = left_y_1 + rotate_item_1.h1 + rotate_item_1.h2
            gap_point_4_x = left_x_1
            gap_point_4_y = left_y_1 + rotate_item_1.h

            gap_point_5_x = left_x_1
            gap_point_5_y = left_y_1 + rotate_item_1.h1
            gap_point_6_x = left_x_1 + rotate_item_1.w2
            gap_point_6_y = left_y_1 + rotate_item_1.h1
            gap_point_7_x = left_x_1 + rotate_item_1.w2
            gap_point_7_y = left_y_1
            gap_point_8_x = left_x_1
            gap_point_8_y = left_y_1
        T_return_list[0].append((gap_point_1_x, gap_point_1_y))
        T_return_list[0].append((gap_point_2_x, gap_point_2_y))
        T_return_list[0].append((gap_point_3_x, gap_point_3_y))
        T_return_list[0].append((gap_point_4_x, gap_point_4_y))
        T_return_list[1].append((gap_point_5_x, gap_point_5_y))
        T_return_list[1].append((gap_point_6_x, gap_point_6_y))
        T_return_list[1].append((gap_point_7_x, gap_point_7_y))
        T_return_list[1].append((gap_point_8_x, gap_point_8_y))
        return T_return_list
    else:
        return '传进来的不是T'


# def is_in_gap(peak_point, gap_point_1, gap_point_2, gap_point_3):
#     x_max = max(gap_point_1[0], gap_point_2[0], gap_point_3[0])
#     x_min = min(gap_point_1[0], gap_point_2[0], gap_point_3[0])
#     y_max = max(gap_point_1[1], gap_point_2[1], gap_point_3[1])
#     y_min = min(gap_point_1[1], gap_point_2[1], gap_point_3[1])
#     if x_min <= peak_point[0] <= x_max and y_min <= peak_point[1] <= y_max:
#         return True
#     else:
#         return False


# 判断一点是否在矩形中,落在边界上也算在矩形中
def is_in_envelope_rectangle(peak_point, envelope_point_1, envelope_point_2, envelope_point_3, envelope_point_4):
    x_max = max(envelope_point_1[0], envelope_point_2[0], envelope_point_3[0], envelope_point_4[0])
    x_min = min(envelope_point_1[0], envelope_point_2[0], envelope_point_3[0], envelope_point_4[0])
    y_max = max(envelope_point_1[1], envelope_point_2[1], envelope_point_3[1], envelope_point_4[1])
    y_min = min(envelope_point_1[1], envelope_point_2[1], envelope_point_3[1], envelope_point_4[1])
    if x_min <= peak_point[0] <= x_max and y_min <= peak_point[1] <= y_max:
        return True
    else:
        return False
