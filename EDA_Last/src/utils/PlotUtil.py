import random
import PyQt5
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QBrush, QPen, QPainter, QPolygon
from PyQt5.QtCore import QPoint, Qt


class Window(QMainWindow):
    def __init__(self, area, result_list):
        super().__init__()

        self.title = "绘制多边形"
        self.area = area
        self.result_list = result_list
        self.InitWindow()

    def InitWindow(self):
        self.setWindowTitle(self.title)
        self.show()
        self.resize(500, 500)

    def plot(self, data, painter, color, brushEnable):
        q_list = []
        w = 8
        b = (20, 20)
        for x, y in data:
            q_list.append(QPoint(x * w + b[0], y * w + b[1]))

        points = QtGui.QPolygon(q_list)
        painter.translate(0, self.height())
        painter.scale(1, -1)
        if brushEnable:
            painter.setBrush(QBrush(Qt.yellow))
        else:
            painter.setBrush(QtGui.QBrush(Qt.NoBrush))
        painter.drawPolygon(points)

    def get_absolute_position_by_left_bottom_point(self, boundary, left_bottom_point):
        for i in range(len(boundary)):
            boundary[i] = [boundary[i][0] + left_bottom_point[0], boundary[i][1] + left_bottom_point[1]]
        return boundary

    def paintEvent(self, event):

        index = 0
        ports_list = []
        s = "data:["
        for result in self.result_list:
            boundary = result.rotate_item.boundary.copy()
            for i in range(len(boundary)):
                x, y = boundary[i]
                boundary[i] = [x + result.center_position[0] - result.rotate_item.w / 2.0,
                               y + result.center_position[1] - result.rotate_item.h / 2.0]
            s = s + str(boundary) + ","
            self.plot(boundary, self.get_random_module_painter(index % 5)[0], Qt.yellow, True)

            ports = []
            for p in result.rotate_item.ports:
                ports.append(self.get_absolute_position_by_left_bottom_point(p.copy(), result.left_bottom_point))
            ports_list.append(ports)

            index += 1
        s += "],"
        print(s)
        print("ports:" + str(ports_list))
        area_painter = QPainter(self)
        area_painter.setPen(QPen(Qt.black, 3, Qt.DotLine))
        self.plot(self.area, area_painter, None, False)

    def get_random_module_painter(self, r):
        if r == 0:
            module_painter = QPainter(self)
            module_painter.setPen(QPen(Qt.red, 3, Qt.SolidLine))
            return module_painter, Qt.red
        elif r == 1:
            module_painter = QPainter(self)
            module_painter.setPen(QPen(Qt.blue, 3, Qt.SolidLine))
            return module_painter, Qt.blue
        elif r == 2:
            module_painter = QPainter(self)
            module_painter.setPen(QPen(Qt.darkYellow, 3, Qt.SolidLine))
            return module_painter, Qt.darkYellow
        elif r == 3:
            module_painter = QPainter(self)
            module_painter.setPen(QPen(Qt.green, 3, Qt.SolidLine))
            return module_painter, Qt.green
        elif r == 4:
            module_painter = QPainter(self)
            module_painter.setPen(QPen(Qt.darkMagenta, 3, Qt.SolidLine))
            return module_painter, Qt.darkMagenta
        else:
            raise RuntimeError("出现了新的颜色！")
