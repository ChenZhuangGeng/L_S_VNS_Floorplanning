from src.classes.EdaClasses import *
import linecache


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


def get_instance(ports_area_input_txt_path, ports_link_input_txt_path):
    link_list = read_data_port_link(ports_link_input_txt_path)
    item_list, rule_dict, area_list = read_data_port_area(ports_area_input_txt_path)
    instance = Instance(item_list, link_list, rule_dict, area_list)
    return instance


def read_data_port_area(ports_area_input_txt_path):
    return_data = []
    area_list = []
    rule_dict = {}
    item_list = []

    file = open(ports_area_input_txt_path, 'r')
    line_1 = linecache.getline(ports_area_input_txt_path, 1).replace(' ', '')
    line_1 = line_1.replace('Area:', '').replace(')', ' ').replace(',', ' ').replace('(', '').replace('\n', '')
    area_point_list = line_1.split(' ')
    area_point_list.pop()
    for i in range(0, len(area_point_list) - 1, 2):
        x = float(area_point_list[i])
        y = float(area_point_list[i + 1])
        area_list.append([x, y])
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
    txt_data = file.readlines()
    txt_data.pop(0)
    txt_data.pop(0)

    for i in range(0, len(txt_data) - 1, 5):
        # name
        item_name = txt_data[i].replace('Module:', '').replace('\n', '')
        # boundary
        item_boundary_list = []
        temp = txt_data[i + 1].replace('Boundary:', '').replace('(', '').replace(')', ' ').replace('\n', '').replace(
            ',', '')
        boundary_layer = txt_data[i + 1].split(';')[1].replace('\n', '')
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
        port_list = [[], [], []]
        port_layers = []
        for k in range(3):
            temp = txt_data[i + 2 + k].replace('Port:', '').replace('(', '').replace(')', ' ').replace('\n',
                                                                                                       '').replace(',',
                                                                                                                   '').split(
                ' ')
            temp.pop()
            port_layers.append(txt_data[i + 2 + k].split(';')[1].replace('\n', ''))
            for m in range(0, len(temp) - 1, 2):
                temp_list = []
                x = float(temp[m])
                y = float(temp[m + 1])
                temp_list.append(x)
                temp_list.append(y)
                port_list[k].append(temp_list)
            port_list[k] = [[x - x_min, y - y_min] for x, y in port_list[k]]
        item = Item(item_name, item_boundary_list, port_list, boundary_layer, port_layers)
        item.init_item()
        item_list.append(item)
    return_data.append(item_list)
    return_data.append(rule_dict)
    return_data.append(area_list)
    return return_data


def read_data_port_link(ports_link_input_txt_path):
    return_data = []
    file = open(ports_link_input_txt_path, 'r')
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
        link = Link(link_dict)
        return_data.append(link)
    file.close()
    return return_data
