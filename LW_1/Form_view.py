import numpy as np
from PyQt5 import QtWidgets, QtCore, uic, QtGui
from PyQt5.QtWidgets import QFileDialog, QTableWidget, QTableWidgetItem
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import sys
import cv2
import qimage2ndarray
import graph as gr


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('Form.ui', self)

        self.graph = gr.Graph(1)
        # Кнопки
        self.Count_points_spinBox.valueChanged.connect(self.change_count_point)
        self.Generate_graph_pushButton.clicked.connect(self.generate_graph)
        self.Save_Graph_pushButton.clicked.connect(self.change_adjacency_table_data)
        self.Back_Graph_pushButton.clicked.connect(lambda: self.create_table(self.graph.adjacency_table))

    def change_count_point(self):
        """Максимальное значение spinbox"""
        max_value = self.Count_points_spinBox.value()
        self.Start_point_spinBox.setMaximum(max_value)
        self.Finish_points_spinBox.setMaximum(max_value)

    def generate_graph(self):
        """Генерация графа"""
        count_point = self.Count_points_spinBox.value()
        self.graph = gr.Graph(count_point)
        table = self.graph.adjacency_table

        self.create_table(table)
        self.show_graph()

    def change_adjacency_table_data(self):
        """Получить значение таблицы"""
        rows = self.Adjacency_TabletableWidget.rowCount()
        columns = self.Adjacency_TabletableWidget.columnCount()

        data = [
            [int(self.Adjacency_TabletableWidget.item(row, column).text())
             for column in range(columns)
             if self.Adjacency_TabletableWidget.item(row, column) is not None
             ]
            for row in range(rows)
        ]

        self.graph.adjacency_table = data

    def create_table(self, adjacency_table):
        """Заполнить таблицу"""
        self.Adjacency_TabletableWidget.setRowCount(len(adjacency_table))
        self.Adjacency_TabletableWidget.setColumnCount(len(adjacency_table[0]))
        self.Adjacency_TabletableWidget.setHorizontalHeaderLabels([str(i + 1) for i in range(len(adjacency_table))])

        for row in range(len(adjacency_table)):
            for column in range(len(adjacency_table[0])):
                item = QTableWidgetItem(str(adjacency_table[row][column]))
                item.setTextAlignment(Qt.AlignHCenter)

                self.Adjacency_TabletableWidget.setItem(row, column, item)

    def show_graph(self):
        if self.Graph_view_checkBox.isChecked() is False:
            self.Image_label.clear()
            return

        path_net = self.graph.draw_graph(show=False)
        self.view_image(path_net)

    def view_image(self, image_path):
        # Считываю изображение по пути
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Изменение размера изображения
        size = self.Image_label.size()
        h, w = size.height(), size.width()
        image = cv2.resize(image, (w, h), cv2.INTER_CUBIC)
        # Перевод изображения в Qpixmap
        image2pixmap = QPixmap.fromImage(qimage2ndarray.array2qimage(image))
        self.Image_label.clear()
        self.Image_label.setPixmap(image2pixmap)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    window.show()
    app.exec_()
