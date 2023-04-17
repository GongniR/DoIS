"""'
Пахомов Д.В.
Группа: 224-321
Почта: pakhomovdv0@gmail.com
Курс: https://online.mospolytech.ru/course/view.php?id=10055
Год разработки: 2023
"""

import random
import matplotlib.pyplot as plt
import networkx as nx
import os


class Graph:
    def __init__(self, count_point, start_point=-1, finish_point=-1):
        self.adjacency_table = self.__get_random_adjacency_table(count_point)
        self.nodes = [i for i in range(len(self.adjacency_table))]
        self.edges = self.__get_edges(self.adjacency_table)
        self.start_point = start_point
        self.finish_point = finish_point
        self.shortest_path = [0]

    @staticmethod
    def __get_random_adjacency_table(count_point: int):
        """Сгенерировать граф смежности """
        rand_list = [i for i in range(1, count_point)] + [100]
        adjacency_table = []

        for i in range(count_point):
            if i == 0:
                adjacency_table.append(random.sample(rand_list, count_point - i))
            else:
                adjacency_table.append(
                    [elem[i] for elem in adjacency_table] + random.sample(rand_list, count_point - i))

        for i in range(count_point):
            adjacency_table[i][i] = 100

        return adjacency_table

    @staticmethod
    def __get_edges(graph):
        """Получить ребра графа"""
        edges = []
        for i in range(len(graph) - 1):
            for j in range(len(graph[i + 1])):
                if graph[i][j] != 100:
                    edges.append((i, j))
        return edges

    def set_adjacency_table(self, adjacency_table):
        self.adjacency_table = adjacency_table
        self.nodes = [i for i in range(len(adjacency_table))]
        self.edges = self.__get_edges(adjacency_table)

    def generate_new_graph(self, count_point):
        adjacency_table = self.__get_random_adjacency_table(count_point)
        self.set_adjacency_table(adjacency_table)

    def set_shortest_path(self, new_path, start_point, finish_point):
        self.start_point = start_point
        self.finish_point = finish_point
        self.shortest_path = new_path

    def __get_shortest_path_edges(self):
        index = self.shortest_path.index(self.finish_point) + 1
        if index == 0:
            path = [self.start_point] + [self.shortest_path[index]]
        else:
            path = [self.start_point] + self.shortest_path[:index]

        edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        return edges

    def draw_graph(self, show=True, path_short=False):
        graph = self.adjacency_table
        nodes = self.nodes
        edges = self.edges

        G = nx.complete_graph(len(nodes))
        pos = nx.spring_layout(G, seed=11)

        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color="tab:blue")
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges,
            width=3,
            alpha=0.5,
            edge_color="tab:red",
        )
        path = 'E:\\GitHub\\DoIS\\LW_1\\plot\\network.png'
        if path_short:
            path_edges = self.__get_shortest_path_edges()
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=path_edges,
                width=8,
                alpha=1,
                edge_color="tab:green",
            )
            path = 'E:\\GitHub\\DoIS\\LW_1\\plot\\network_path.png'
        labels = {i: str(i + 1) for i in range(len(nodes))}
        nx.draw_networkx_labels(G, pos, labels, font_size=15, font_color="whitesmoke")

        # рисуем граф и отображаем его
        plt.tight_layout()
        plt.axis("off")
        if show:
            plt.show()
            plt.close()
        else:
            try:
                os.remove(path)
            except:
                pass

            plt.savefig(path)
            plt.close()
            return path
