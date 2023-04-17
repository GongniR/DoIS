"""'
Пахомов Д.В.
Группа: 224-321
Почта: pakhomovdv0@gmail.com
Курс: https://online.mospolytech.ru/course/view.php?id=10055
Год разработки: 2023
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore, uic, QtGui
from PyQt5.QtWidgets import QFileDialog, QTableWidget, QTableWidgetItem
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import sys
import cv2
import qimage2ndarray
import graph as gr
import subprocess
import genetic_algorithm as ga


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('Form.ui', self)

        self.graph = gr.Graph(1)
        self.gen_alg = 0
        # Кнопки
        # pushButton
        self.path_short = False
        # граф
        self.Count_points_spinBox.valueChanged.connect(self.change_count_point)
        self.Generate_graph_pushButton.clicked.connect(self.generate_graph)
        self.Save_Graph_pushButton.clicked.connect(self.change_adjacency_table_data)
        self.Back_Graph_pushButton.clicked.connect(lambda: self.create_table(self.graph.adjacency_table))
        #  ГА
        self.Create_GA_pushButton.clicked.connect(self.create_population)
        self.Train_GA_pushButton.clicked.connect(self.Train_GA)
        self.Train_Step_pushButton.clicked.connect(self.step_ga)

        # radioButton
        self.Graph_view_radioButton.toggled.connect(self.show_graph)
        self.Statistics_view_radioButton.toggled.connect(self.show_statistics)
        self.Graph_radioButton.toggled.connect(lambda: self.create_table(self.graph.adjacency_table))
        self.Ind_radioButton.toggled.connect(self.get_table_population)
        self.Generation_radioButton.toggled.connect(self.get_table_generation)


    def create_population(self):
        size_population = self.Size_population_spinBox.value()
        count_generation = self.Count_General_spinBox.value()
        mutation = self.Mutantion_doubleSpinBox.value() / 100
        start_point = self.Start_point_spinBox.value()-1
        finish_point = self.Finish_points_spinBox.value()-1

        self.gen_alg = ga.GA(adjacency_table=self.graph.adjacency_table,
                             start_point=start_point,
                             finish_point=finish_point,
                             size_population=size_population,
                             mutation=mutation,
                             max_generation=count_generation)

        self.gen_alg.createGA()

    def get_table_population(self):
        self.Adjacency_TabletableWidget.setRowCount(len(self.gen_alg.population))
        self.Adjacency_TabletableWidget.setColumnCount(len(self.gen_alg.population[0]))
        self.Adjacency_TabletableWidget.setHorizontalHeaderLabels(
            [str(i + 1) for i in range(len(self.gen_alg.population[0]))])

        for row in range(len(self.gen_alg.population)):
            for column in range(len(self.gen_alg.population[0])):
                item = QTableWidgetItem(str(self.gen_alg.population[row][column]))
                item.setTextAlignment(Qt.AlignHCenter)

                self.Adjacency_TabletableWidget.setItem(row, column, item)

    def get_table_generation(self):
        self.Adjacency_TabletableWidget.setRowCount(len(self.gen_alg.list_population[0]))
        self.Adjacency_TabletableWidget.setColumnCount(len(self.gen_alg.list_population))
        self.Adjacency_TabletableWidget.setHorizontalHeaderLabels(
            [str(f'Поколение {i}') for i in range(len(self.gen_alg.list_population))])

        for row in range(len(self.gen_alg.list_population[0])):
            for column in range(len(self.gen_alg.list_population)):
                item = QTableWidgetItem(str(self.gen_alg.list_population[column][row]))
                item.setTextAlignment(Qt.AlignHCenter)

                self.Adjacency_TabletableWidget.setItem(row, column, item)

    def Train_GA(self):
        self.gen_alg.TrainGA()
        self.Statistics_view_radioButton.setChecked(True)
        self.show_statistics()
        self.show_generation_info()
        self.draw_shortest_path()
    def change_count_point(self):
        """Максимальное значение spinbox"""
        max_value = self.Count_points_spinBox.value()
        self.Start_point_spinBox.setMaximum(max_value)
        self.Finish_points_spinBox.setMaximum(max_value)

    def generate_graph(self):
        self.graph.shortest_path = False
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
        self.graph.path_short = False
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
        if self.Graph_view_radioButton.isChecked() is False:
            self.Image_label.clear()
            return

        path_net = self.graph.draw_graph(show=False, path_short=self.path_short)
        self.view_image(path_net)

    def draw_shortest_path(self):
        path = self.gen_alg.get_shortest_path()
        start_point = self.Start_point_spinBox.value()-1
        finish_point = self.Finish_points_spinBox.value()-1
        self.graph.set_shortest_path(new_path=path,
                                     start_point=start_point,
                                     finish_point=finish_point)
        self.path_short = True

    def show_statistics(self):
        if self.Statistics_view_radioButton.isChecked() is False:
            self.Image_label.clear()
            return

        path_state = self.gen_alg.draw_statistics()
        self.view_image(path_state)

    def show_generation_info(self):
        for line in self.gen_alg.statistics_generation:
            text = self.Log_textEdit.toPlainText() + line + '\n'
            self.Log_textEdit.setPlainText(text)

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

    def step_ga(self):
        self.gen_alg.checkStep+=1
        self.gen_alg.StepGA()
        text = self.Log_textEdit.toPlainText() + self.gen_alg.log_step + '\n'
        self.Log_textEdit.setPlainText(text)
        self.get_table_population()

if __name__ == '__main__':
    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'): QtWidgets.QApplication.setAttribute(
        QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'): QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps,
                                                                                       True)
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    window.show()
    app.exec_()
