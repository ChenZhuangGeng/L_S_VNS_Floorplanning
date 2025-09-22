from enum import Enum
import src.function.CzgFunction as CzgFunction
import src.function.WskhFunction as WskhFunction


# 旋转类型枚举类
class Orient(Enum):
    R0 = "R0"
    R90 = "R90"
    R180 = "R180"
    R270 = "R270"
    MX = "MX"
    MY = "MY"
    MXR90 = "MXR90"
    MYR90 = "MYR90"


# 模块类
class Item:
    def __init__(self, name, boundary, ports, boundary_layer, port_layers):
        self.name = name  # 模块名（字符串）
        self.boundary = boundary  # 模块边界顶点坐标列表（列表）
        self.ports = ports  # 模块的端口列表（列表）
        self.boundary_layer = boundary_layer
        self.port_layers = port_layers

    def to_string(self):
        return 'Item', {
            'name': self.name,
            'boundary': self.boundary,
            'ports': self.ports,
            'area': self.area,
            'boundary_layer': self.boundary_layer,
            'port_layers': self.port_layers,
        }
    def to_stringbyczg(self):
        return 'Item', {
            'name': self.name,
            'boundary': self.boundary,
            'ports': self.ports,
            'boundary_layer': self.boundary_layer,
            'port_layers': self.port_layers,
        }

    # 初始化模块信息
    def init_item(self):
        # 计算模块的面积
        self.w, self.h = WskhFunction.calc_envelope_rectangle_w_h(self.boundary)
        self.area = WskhFunction.calc_area(self.w, self.h, self.boundary)
        # 计算每一种旋转状态下的RotateItem
        self.rotate_items = []
        for orient_value, orient in Orient.__members__.items():
            # if orient is not Orient.R270:
            #     continue
            w, h, boundary, orient, ports = CzgFunction.calc_rotate_item(self.w, self.h, self.boundary.copy(),
                                                                         self.ports.copy(), orient)
            rotate_item = RotateItem(self.name, w, h, boundary, orient, ports)
            rotate_item.init_rotate_item()
            self.rotate_items.append(rotate_item)

    @staticmethod
    def copy(item):
        if isinstance(item, Item):
            copy_item = Item(item.name, item.boundary.copy(), item.ports.copy(), item.boundary_layer,
                             item.port_layers.copy())
            copy_item.area = item.area
            copy_item.w = item.w
            copy_item.h = item.h
            copy_item.rotate_items = item.rotate_items.copy()
            return copy_item
        else:
            arr = []
            for i in item:
                arr.append(Item.copy(i))
            return arr


# 旋转模块类
class RotateItem:
    def __init__(self, name, w, h, boundary, orient, ports):
        self.name = name
        self.boundary = boundary
        self.orient = orient
        self.w = w
        self.h = h
        self.ports = ports

    def to_string(self):
        return 'RotateItem', {
            'name': self.name,
            'w': self.w,
            'h': self.h,
            'orient': self.orient,
            'boundary': self.boundary,
            'ports': self.ports,
        }

    def init_rotate_item(self):
        # 计算宽和高
        self.w, self.h = WskhFunction.calc_envelope_rectangle_w_h(self.boundary)
        # 计算所有竖着线段(x,y,l)
        self.vertical_line_list = WskhFunction.calc_vertical_line_list(self.boundary)
        # 计算所有水平线段(x,y,l)
        self.horizontal_line_list = WskhFunction.calc_horizontal_line_list(self.boundary)
        # 获取着地水平线段(x,y,l)
        for horizontal_line in self.horizontal_line_list:
            if horizontal_line[1] == 0:
                self.floor_horizontal_line = horizontal_line
                break
        # 计算放入当前模块将会新增的天际线(x,y,l)
        self.add_sky_line_list = WskhFunction.calc_add_sky_line_list(self.horizontal_line_list)
        # 找到第二高度
        self.sec_y = None
        for x, y in self.boundary:
            if y != 0 and (self.sec_y is None or self.sec_y > y):
                self.sec_y = y
        # 计算对标点
        self.aim_point_list = WskhFunction.calc_aim_point_list(self.boundary)
        # 获取当前模块type
        if len(self.boundary) == 4:
            self.rect_list = [[0, 0, self.w, self.h]]
        elif len(self.boundary) == 6:
            # L型
            if len(self.add_sky_line_list) == 1:
                if len(self.aim_point_list) == 2:
                    self.type = '180'
                    self.w2 = self.floor_horizontal_line[2]
                    self.w1 = self.w - self.w2
                    self.h2 = self.sec_y
                    self.h1 = self.h - self.h2
                    self.rect_list = [[0, self.h2, self.w1, self.h1], [self.w1, 0, self.w2, self.h]]
                elif len(self.aim_point_list) == 1:
                    self.type = '90'
                    self.w1 = self.floor_horizontal_line[2]
                    self.w2 = self.w - self.w1
                    self.h1 = self.sec_y
                    self.h2 = self.h - self.h1
                    self.rect_list = [[0, 0, self.w1, self.h], [self.w1, self.h1, self.w2, self.h2]]
            elif len(self.add_sky_line_list) == 2:
                # 找到(0,0)这个点，通过判断其两边的情况，判断是0度还是270度
                i = 0
                while i < 6:
                    if self.boundary[i][0] == 0 and self.boundary[i][1] == 0:
                        i_f = i - 1 if i - 1 >= 0 else 5
                        i_b = i + 1 if i + 1 < 6 else 0
                        if self.boundary[i_f][1] == self.h or self.boundary[i_b][1] == self.h:
                            self.type = '0'
                            for add_sky_line in self.add_sky_line_list:
                                if add_sky_line[0] == 0:
                                    self.w1 = add_sky_line[2]
                                elif add_sky_line[0] + add_sky_line[2] == self.w:
                                    self.w2 = add_sky_line[2]
                                    self.h2 = add_sky_line[1]
                                    self.h1 = self.h - self.h2
                            self.rect_list = [[0, 0, self.w1, self.h], [self.w1, 0, self.w2, self.h2]]
                        else:
                            self.type = '270'
                            for add_sky_line in self.add_sky_line_list:
                                if add_sky_line[0] == 0:
                                    self.w1 = add_sky_line[2]
                                    self.h1 = add_sky_line[1]
                                    self.h2 = self.h - self.h1
                                elif add_sky_line[0] + add_sky_line[2] == self.w:
                                    self.w2 = add_sky_line[2]
                            self.rect_list = [[0, 0, self.w1, self.h1], [self.w1, 0, self.w2, self.h]]
                        break
                    i += 1
        elif len(self.boundary) == 8:
            # T型
            if len(self.add_sky_line_list) == 1:
                self.type = '180'
                self.w2 = self.floor_horizontal_line[2]
                for horizontal_line in self.horizontal_line_list:
                    if horizontal_line[0] == 0 and horizontal_line[2] < self.w:
                        self.w1 = horizontal_line[2]
                        self.h2 = self.h - horizontal_line[1]
                        self.h1 = self.h - self.h2
                    elif horizontal_line[0] != 0 and horizontal_line[0] + horizontal_line[2] == self.w:
                        self.w3 = horizontal_line[2]
                        self.h3 = horizontal_line[1]
                        self.h4 = self.h - self.h3
                self.rect_list = [[0, self.h2, self.w1, self.h], [self.w1, 0, self.w2, self.h],
                                  [self.w1 + self.w2, self.h3, self.w3, self.h4]]
            elif len(self.add_sky_line_list) == 2:
                if len(self.aim_point_list) == 2:
                    self.type = '270'
                    self.w1 = self.floor_horizontal_line[2]
                    for add_sky_line in self.add_sky_line_list:
                        if add_sky_line[0] == 0:
                            self.w3 = add_sky_line[2]
                            self.h3 = self.h - add_sky_line[1]
                        else:
                            self.w4 = add_sky_line[2]
                    for horizontal_line in self.horizontal_line_list:
                        if horizontal_line[0] == 0 and horizontal_line[1] < self.h - self.h3:
                            self.w2 = horizontal_line[2]
                            self.h1 = horizontal_line[1]
                            self.h2 = self.h - self.h1 - self.h3
                            break
                    self.rect_list = [[self.w2, 0, self.w1, self.h1], [0, self.h1, self.w, self.h2],
                                      [self.w3, self.h1 + self.h2, self.w4, self.h3]]
                elif len(self.aim_point_list) == 1:
                    self.type = '90'
                    self.w1 = self.floor_horizontal_line[2]
                    for add_sky_line in self.add_sky_line_list:
                        if add_sky_line[0] == 0:
                            self.w4 = add_sky_line[2]
                        else:
                            self.w3 = add_sky_line[2]
                            self.h1 = self.h - add_sky_line[1]
                    for horizontal_line in self.horizontal_line_list:
                        if horizontal_line[0] + horizontal_line[2] == self.w and horizontal_line[1] < self.h - self.h1:
                            self.w2 = horizontal_line[2]
                            self.h3 = horizontal_line[1]
                            self.h2 = self.h - self.h1 - self.h3
                            break
                    self.rect_list = [[0, 0, self.w1, self.h3], [0, self.h3, self.w, self.h2],
                                      [0, self.h2 + self.h3, self.w4, self.h1]]
            elif len(self.add_sky_line_list) == 3:
                self.type = '0'
                for asl in self.add_sky_line_list:
                    if asl[0] == 0:
                        self.h1 = asl[1]
                        self.w1 = asl[2]
                        self.h2 = self.h - self.h1
                    elif asl[0] + asl[2] == self.w:
                        self.h4 = asl[1]
                        self.w3 = asl[2]
                        self.h3 = self.h - self.h4
                self.w2 = self.w - self.w1 - self.w3
                self.rect_list = [[0, 0, self.w1, self.h1], [self.w1, 0, self.w2, self.h],
                                  [self.w1 + self.w2, 0, self.w3, self.h4]]


# 连接类
class Link:
    def __init__(self, link_dict):
        self.link_dict = link_dict  # 连接字典（字典）

    def to_string(self):
        return 'Link', {
            'link_dict': self.link_dict,
        }

    @staticmethod
    def copy(link):
        if isinstance(link, Link):
            return Link(link.link_dict.copy())
        else:
            arr = []
            for i in link:
                arr.append(Link.copy(i))
            return arr


# 实例类
class Instance:
    def __init__(self, items, links, rule, area):
        self.items = items  # 模块列表（列表）
        self.links = links  # 连接列表（列表）
        self.rule = rule  # 规则字典（字典）
        self.area = area  # 布线区域（顶点坐标列表）

    def init_instance(self):
        pass

    @staticmethod
    def copy(instance):
        return Instance(Item.copy(instance.items), Link.copy(instance.links), instance.rule.copy(),
                        instance.area.copy())


# 评价结果类
class Evaluation:
    def __init__(self, obj_value, sequence, result_list):
        self.obj_value = obj_value
        self.sequence = sequence
        self.result_list = result_list

    @staticmethod
    def copy(evaluation):
        return Evaluation(evaluation.obj_value, evaluation.sequence.copy(), Result.copy(evaluation.result_list))


# 结果类
class Result:
    def __init__(self, item, rotate_item, orient, center_position, sky_line, left_bottom_point):
        self.item = item  # 模块
        self.rotate_item = rotate_item  # 旋转后的模块
        self.orient = orient  # 模块旋转类型
        self.center_position = center_position  # 模块中心点坐标
        self.sky_line = sky_line  # 模块放置的天际线
        self.left_bottom_point = left_bottom_point  # 模块放置的对标点

    def to_string(self):
        return 'Result', {
            'item': self.item.to_string(),
            'orient': self.orient,
            'center_position': self.center_position,
            'sky_line': self.sky_line,
            'left_bottom_point': self.left_bottom_point,
        }

    @staticmethod
    def copy(result):
        if isinstance(result, Result):
            return Result(Item.copy(result.item), result.rotate_item, result.orient,
                          result.center_position.copy(), result.sky_line, result.left_bottom_point)
        else:
            arr = []
            for i in result:
                arr.append(Result.copy(i))
            return arr


# MyResult对象
class MyResult:
    def __init__(self, best_obj_value, timer, accuracy, epochs_size, fit_cnt, predict_cnt, predict_true_cnt,
                 result_list_len):
        self.best_obj_value = best_obj_value
        self.timer = timer
        self.accuracy = accuracy
        self.epochs_size = epochs_size
        self.fit_cnt = fit_cnt
        self.predict_cnt = predict_cnt
        self.predict_true_cnt = predict_true_cnt
        self.result_list_len = result_list_len
