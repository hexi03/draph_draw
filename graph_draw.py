import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMenu
from PyQt5.QtGui import QPainter, QColor, QPen, QFontMetrics, QImage
from PyQt5.QtCore import Qt
from collections import deque
from accessify import private, protected
from typing import Callable, Optional
import math


class NodeDefault():
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.init_ui()

    def init_ui(self):
        self.R = 10
        self.R_scaled = 20
        self.x, self.y = 0, 0
        self.x_scaled, self.y_scaled = 0, 0
        self.scale = 1
        self._is_visible = True
        self.image = None
        self.text = ""
        self._is_selected=False
        self.main_color = QColor("#FF6200")#QColor(255, 0, 0)
        self.second_color = QColor("#6BEC3B")#QColor(0, 255, 0)
        self.selected_color=QColor("#4380D3")#QColor(255,0,255)
        self.text_color = QColor(0, 0, 200)#self.second_color


    def get_id(self) -> str:
        return self.node_id

    def set_size(self, r: float):
        self.R = r

    def set_visibility(self, f: bool):
        self._is_visible = f

    def is_visible(self) -> bool:
        return self._is_visible

    def set_selection(self,f:bool):
        self._is_selected=f
    def is_selected(self)->bool:
        return self._is_selected

    def get_size(self) -> float:
        return self.R_scaled

    def get_position(self) -> list:
        return [self.x_scaled, self.y_scaled]

    def get_real_position(self) -> list:
        return [self.x, self.y]

    def set_position(self, x: float, y: float):
        self.x, self.y = x, y
        self.x_scaled = self.x * self.scale
        self.y_scaled = self.y * self.scale

    def set_scaling(self, scale: float):
        self.scale = scale
        self.x_scaled = self.x * self.scale
        self.y_scaled = self.y * self.scale
        self.R_scaled = self.R * self.scale

    def set_image(self, img: QImage):
        self.image = img

    def set_text(self, text: str):
        self.text = text
    def get_text(self):
        return self.text

    def set_main_color(self, color: QColor):
        self.main_color = color

    def set_second_color(self, color: QColor):
        self.second_color = color
        r,g,b=color.getRgb()
        r=255-r
        g=255-g
        b=255-b
        self.selected_color=QColor(r,g,b)

    def set_text_color(self, color: QColor):
        self.text_color = color

    def add_bias(self, vx: float, vy: float):
        # self.set_position()
        self.x_scaled += vx
        self.y_scaled += vy
        self.x += vx / self.scale
        self.y += vy / self.scale

    def draw_node(self, qp: QPainter):
        if not self.is_visible():
            return
        pen = QPen()
        pen.setWidth(3)
        if self.is_selected():
            pen.setColor(self.selected_color)
        else:
            pen.setColor(self.second_color)
        qp.setPen(pen)
        qp.setBrush(self.main_color)
        if self.image == None:
            qp.drawEllipse(self.x_scaled-self.R_scaled, self.y_scaled-self.R_scaled, self.R_scaled * 2, self.R_scaled * 2)
        else:
            imbuf = self.image.scaled(self.R_scaled * 2-10, self.R_scaled * 2-10)
            qp.drawRect(self.x_scaled-self.R_scaled
                        ,self.y_scaled-self.R_scaled
                        ,self.R_scaled*2,self.R_scaled*2)
            qp.drawImage(self.x_scaled+5-self.R_scaled, self.y_scaled+5-self.R_scaled, imbuf)



    def draw_node_text(self, qp: QPainter):
        if not self.is_visible():
            return
        pen = QPen()
        # pen.setWidth(self.width)
        pen.setColor(self.text_color)
        qp.setPen(pen)
        width = QFontMetrics(qp.font()).width(self.text) / 2
        qp.drawText(self.x_scaled - width, self.y_scaled + self.R_scaled + 20, self.text)

    def is_point_belongs_to_node(self, x: float, y: float) -> bool:
        if self.image == None:
            xd, yd = abs(x - self.x_scaled), abs(
                y - self.y_scaled)
            if self.R_scaled >= int(((xd ** 2) + (yd ** 2)) ** 0.5):
                return True
            return False
        else:
            xd, yd = self.x_scaled, self.y_scaled
            rd = self.R_scaled
            if xd-rd <= x <= xd + rd:
                if yd + rd >= y >= yd-rd:
                    return True
            return False


class EdgeDefault():
    def __init__(self, a: NodeDefault, b: NodeDefault):
        self.node_a, self.node_b = a, b
        self.value = 0
        self.init_ui()

    def init_ui(self):
        self.width = 10
        self.color = QColor("#0F4FA8")#QColor(200, 0, 0)
        self.text_color = QColor(0, 200, 0)
        self.back_color = QColor("#3BDA00")#QColor(0, 0, 0)
        self._is_visible = True

        #self.font = open('fonts/Awesome.otf').read()

    def set_visibility(self, f: bool):
        self._is_visible = f

    def is_visible(self) -> bool:
        return self._is_visible

    def set_width(self, w: float):
        self.width = w

    def set_value(self, v: str):
        self.value = v

    """
    def set_value(self, v: int):
        self.value = v

    def set_value(self, v: float):
        self.value = v
    """

    def get_value(self) -> str:
        return str(self.value)

    """
    def get_value(self) -> int:
        return self.value

    def get_value(self) -> float:
        return self.value
    """

    def get_nodes(self) -> list:
        return [self.node_a, self.node_b]

    def get_positions_of_nodes(self) -> list:
        return [self.node_a.get_position(), self.node_b.get_position()]

    def is_point_belongs_to_edge(self, x: float, y: float) -> bool:
        a_p, b_p = self.get_positions_of_nodes()
        xa, ya = a_p
        xb, yb = b_p
        """
        if xa <= x <= xb or xb <= x <= xa:
            if ya <= y <= yb or yb <= y <= ya:
            """
        #print(self.node_a.get_id(),self.node_b.get_id())
        l = self.width/2

        if xb==xa:
            if abs(x-xa)<=l and (ya <= y <= yb or yb <= y <= ya):
                return True
            else:
                return False
        if ya==yb:
            if abs(y-ya)<=l and (xa <= x <= xb or xb <= x <= xa):
                return True
            else:
                return False
        if not ((xa <= x <= xb or xb <= x <= xa) and (ya <= y <= yb or yb <= y <= ya)):
            return False
        #l = ((((m * 2) ** 2) / 2) ** 0.5) / 2

        k = (yb - ya) / (xb - xa)
        b = ya - (k * xa)
        a = math.atan(k)
        l0=(y-k*x-b)/(k*math.sin(a)+math.cos(a))
        if b-l*math.cos(a)<=y-k*(x+l0*math.sin(a))<=b+l*math.cos(a):
            return True
        else:
            return False
    def set_color(self, color: QColor):
        self.color = color

    def set_text_color(self, color: QColor):
        self.text_color = color

    def draw_edge(self, qp: QPainter):
        if not (self.is_visible() and self.node_a.is_visible() and self.node_b.is_visible()):
            return
        pen = QPen()
        pen.setWidth(self.width)
        pen.setColor(self.back_color)
        qp.setPen(pen)
        a_p, b_p = self.get_positions_of_nodes()
        qp.drawLine(a_p[0] , a_p[1] , b_p[0] , b_p[1] )

        pen = QPen()
        pen.setWidth(self.width-3)
        pen.setColor(self.color)
        qp.setPen(pen)
        qp.drawLine(a_p[0] , a_p[1] , b_p[0] , b_p[1] )

    def draw_edge_text(self, qp: QPainter):
        if not (self.is_visible() and self.node_a.is_visible() and self.node_b.is_visible()):
            return
        pen = QPen()
        # pen.setWidth(self.width)
        pen.setColor(self.color)
        qp.setPen(pen)
        pos1 = self.node_a.get_position()
        pos2 = self.node_b.get_position()
        R1, R2 = self.node_a.get_size(), self.node_b.get_size()
        v1 = (pos2[0]) - (pos1[0]), (pos2[1]) - (pos1[1])
        l = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
        L = (R1 + R2) / 2 + self.width
        if l != 0:
            v2 = v1[1] * (L / l), -v1[0] * (L / l)
            p = ((pos2[0]) + (pos1[0])) / 2 + v2[0], ((pos2[1]) + (pos1[1])) / 2 + v2[1]
            st = str(self.value)
            width = QFontMetrics(qp.font()).width(st) / 2
            qp.drawText(p[0] - width, p[1], st)


class Graph_itty(QtWidgets.QFrame):
    def __init__(self):
        self.n = 0
        self.nodes = deque()
        self.edges = deque()
        self.id_index = dict()
        self.matrix = np.array([[]])
        self.inc_matrix = np.array([[]])
        self.init_ui()

    def init_ui(self):

        self.screen_scale_cntr = 0
        self.event_flag = [False, 0, 0, 0]
        self.render_function = self.default_pushed_square_render
        self.aa_level = 0
        self.pi=3.1415925
        super().__init__()
        self.setStyleSheet("background-color:#FFA873")


    def add_node(self, id: str, **opts: Optional):
        assert id not in self.id_index
        self.n += 1

        self.id_index[id] = self.n - 1
        # --------------------
        new_matrix = np.zeros(self.n ** 2).reshape(self.n, self.n)
        new_matrix[:-1, :-1] = self.matrix[:, :]
        self.matrix = new_matrix
        # -------------------
        new_inc_matrix = np.zeros(self.n ** 2).reshape(self.n, self.n)
        new_inc_matrix[:-1, :-1] = self.inc_matrix[:, :]
        self.inc_matrix = new_inc_matrix
        # --------------------
        if opts.get("node_class"):
            node = opts.get("node_class")(id)
        else:
            node = NodeDefault(id)
        if opts.get("size"):
            node.set_size(opts.get("size"))

        if opts.get("main_color"):
            node.set_main_color(opts.get("main_color"))

        if opts.get("second_color"):
            node.set_second_color(opts.get("second_color"))

        if opts.get("text"):
            node.set_text(opts.get("text"))

        if opts.get("text_color"):
            node.set_text_color(opts.get("text_color"))

        if "image" in dict(opts.items()).keys():
            if type(opts.get("image")) == type(QImage):
                node.set_image(opts.get("image"))
            else:
                Img = opts.get("image")
                height, width, channel = Img.shape
                bytesPerLine = 3 * width
                qImg = QImage(Img.data, width, height, bytesPerLine, QImage.Format_RGB888)
                node.set_image(qImg)
        self.nodes.appendleft(node)
        self.position_render()
        self.repaint()

    def add_nodes(self, ids: list, **opts: Optional):
        li = ids
        # -----------------------
        li = list(set(li))
        li1 = []
        for i in li:
            if i not in self.id_index.keys():
                li1.append(i)
        li = li1
        # -----------------------
        li_len = len(li)
        if li_len == 0:
            return
        self.n += li_len
        # ------------------
        new_matrix = np.zeros(self.n ** 2).reshape(self.n, self.n)
        print(new_matrix.shape, self.matrix.shape, self.n, li_len)
        new_matrix[:-li_len, :-li_len] = self.matrix[:, :]
        self.matrix = new_matrix
        # -------------------
        new_inc_matrix = np.zeros(self.n ** 2).reshape(self.n, self.n)
        new_inc_matrix[:-li_len, :-li_len] = self.inc_matrix[:, :]
        self.inc_matrix = new_inc_matrix
        # -------------------

        for i in range(len(li)):
            j = self.n - li_len + i - 1 + 1
            self.id_index[li[i]] = j

            if opts.get("node_classes"):
                if opts.get("node_classes")[i]:
                    node = opts.get("node_classes")[i](li[i])
                else:
                    node = NodeDefault(li[i])
            else:
                node = NodeDefault(li[i])
            # ---------------------
            if opts.get("sizes"):
                if opts.get("sizes")[i]:
                    node.set_size(opts.get("sizes")[i])

            if opts.get("main_colors"):
                if opts.get("main_colors")[i]:
                    node.set_main_color(opts.get("main_colors")[i])

            if opts.get("second_colors"):
                if opts.get("second_colors")[i]:
                    node.set_second_color(opts.get("second_colors")[i])

            if opts.get("texts"):
                if opts.get("texts")[i]:
                    node.set_text(opts.get("texts")[i])

            if opts.get("text_colors"):
                if opts.get("text_colors")[i]:
                    node.set_text_color(opts.get("text_colors")[i])
            if opts.get("images"):
                if type(opts.get('images')[i]) != type(None):
                    if type(opts.get('images')[i]) == type(QImage):
                        node.set_image(opts.get('images')[i])
                    else:
                        Img = opts.get("images")
                        height, width, channel = Img.shape
                        bytesPerLine = 3 * width
                        qImg = QImage(Img.data, width, height, bytesPerLine, QImage.Format_RGB888)
                        node.set_image(qImg)
            # ---------------------
            self.nodes.appendleft(node)
        self.position_render()
        self.repaint()

    def remove_node(self, id: str):
        a = id
        self.n -= 1
        ai = self.id_index[a]
        # self.nodes.pop(ai)
        for nn in range(self.n):
            node = self.nodes.pop()
            i = node.get_id()
            if i == a:
                pass
            else:
                self.nodes.appendleft(node)
        # ------------------
        self.matrix[ai, :], self.matrix[:, ai] = self.matrix[-1, :], self.matrix[:, -1]
        new_matrix = np.zeros(self.n ** 2).reshape(self.n, self.n)
        new_matrix[:, :] = self.matrix[:-1, :-1]
        self.matrix = new_matrix
        # -------------------
        self.inc_matrix[ai, :], self.inc_matrix[:, ai] = self.inc_matrix[-1, :], self.inc_matrix[:, -1]
        new_inc_matrix = np.zeros(self.n ** 2).reshape(self.n, self.n)
        new_inc_matrix[:, :] = self.inc_matrix[:-1, :-1]
        self.inc_matrix = new_inc_matrix
        # -------------------
        for i in self.id_index.keys():
            if self.id_index[i] == self.n:
                self.id_index[i] = ai
        self.id_index.pop(a)
        self.position_render()
        self.repaint()

    def remove_nodes(self, ids: list):
        li = ids
        li_len = len(li)
        self.n -= li_len
        # ------------------
        for i in range(li_len):
            ai = self.id_index[li[i]]
            self.matrix[ai, :], self.matrix[:, ai] = self.matrix[-1 - i, :], self.matrix[:, -1 - i]
        new_matrix = np.zeros(self.n ** 2).reshape(self.n, self.n)
        new_matrix[:, :] = self.matrix[:-li_len, :-li_len]
        self.matrix = new_matrix
        # -------------------
        for i in range(li_len):
            ai = self.id_index[li[i]]
            self.inc_matrix[ai, :], self.inc_matrix[:, ai] = self.inc_matrix[-1 - i, :], self.inc_matrix[:, -1 - i]
        new_inc_matrix = np.zeros(self.n ** 2).reshape(self.n, self.n)
        new_inc_matrix[:, :] = self.inc_matrix[:-li_len, :-li_len]
        self.inc_matrix = new_inc_matrix
        # -------------------
        # Не эффективно!!!
        # Может не работать
        for nn in range(self.n):
            node = self.nodes.pop()
            i = node.get_id()
            if i in li:
                pass
            else:
                self.nodes.appendleft(node)
        for i in range(li_len):
            ai = self.id_index[li[i]]
            # self.nodes.pop(ai)
            for nn in range(self.n):
                node = self.nodes.pop()
                i = node.get_id()
                if self.id_index[i] == self.n - i:
                    self.id_index[i] = ai

                    self.nodes.appendleft(node)
                    break
                self.nodes.appendleft(node)
            self.id_index.pop(li[i])
        # self.id_index=self.id_index[:-li_len]
        # Не эффективно!!!
        self.position_render()
        self.repaint()

    def add_edge(self, id1: str, id2: str, **opts: Optional):
        a, b = id1, id2
        if a not in self.id_index.keys():
            self.add_node(a)
        if b not in self.id_index.keys():
            self.add_node(b)
        self.inc_matrix[self.id_index[a]][self.id_index[b]] = 1
        if opts.get("value"):
            self.matrix[self.id_index[a]][self.id_index[b]] = opts.get("value")
        # Заменить
        for nn in range(self.n):
            node = self.nodes.pop()
            if a == node.get_id():
                a = node
            if b == node.get_id():
                b = node
            self.nodes.appendleft(node)
        if opts.get("edge_class"):
            edge = opts.get("edge_class")(a, b)
        else:
            edge = EdgeDefault(a, b)
        if opts.get("value"):
            edge.set_value(opts.get("value"))
        self.edges.appendleft(edge)
        self.repaint()

    def add_edges(self, id_pairs: list, **opts: Optional):
        edges = id_pairs
        al, bl = [], []
        for i in range(len(edges)):
            a, b = edges[i]
            if a not in self.id_index.keys():
                al.append(a)
            if b not in self.id_index.keys():
                bl.append(b)
        self.add_nodes(list(set(al + bl)))
        for i in range(len(edges)):
            a, b = edges[i]
            self.inc_matrix[self.id_index[a]][self.id_index[b]] = 1
            # self.inc_matrix[b][a] = 1
            if opts.get("values"):
                if opts.get("values")[i]:
                    self.matrix[self.id_index[a]][self.id_index[b]] = opts.get("values")[i]
                    # self.matrix[b][a] = opts.get("value")[i]
            # Заменить
            for nn in range(self.n):
                node = self.nodes.pop()
                if a == node.get_id():
                    a = node
                if b == node.get_id():
                    b = node
                self.nodes.appendleft(node)
            if opts.get("edge_class"):
                if opts.get("edge_class")[i]:
                    edge = opts.get("edge_class")[i](a, b)
                else:
                    edge = EdgeDefault(a, b)
            else:
                edge = EdgeDefault(a, b)
            if opts.get("values"):
                if opts.get("values")[i]:
                    edge.set_value(opts.get("values")[i])
            self.edges.appendleft(edge)
        self.repaint()

    def remove_edge(self, id1: str, id2: str):
        a, b = id1, id2
        self.inc_matrix[self.id_index[a]][self.id_index[b]] = 0
        # Заменить
        for i in range(len(self.edges)):
            edge = self.edges.pop()
            a_node, b_node = edge.get_nodes()
            a_id, b_id = a_node.get_id(), b_node.get_id()
            if not (a == a_id and b == b_id):
                self.edges.appendleft(edge)
        self.repaint()

    def remove_edges(self, id_pairs: list):
        li = id_pairs
        for a, b in li:
            self.inc_matrix[self.id_index[a]][self.id_index[b]] = 0
            # Заменить
            for i in range(len(self.edges)):
                edge = self.edges.pop()
                a_node, b_node = edge.get_nodes()
                a_id, b_id = a_node.get_id(), b_node.get_id()
                if not (a == a_id and b == b_id):
                    self.edges.appendleft(edge)
        self.repaint()

    def node_clicked_event(self, id: str):
        print(id, "node clicked")

    def edge_clicked_event(self, id1: str, id2: str):
        print(id1, "---", id2, "edge clicked")

    # @protected
    def mousePressEvent(self, e):
        x = e.x()
        y = e.y()
        btns = e.buttons()
        if btns == Qt.RightButton:
            buf_deq=deque()
            for nn in range(self.n):
                node = self.nodes.pop()
                if node.is_point_belongs_to_node(x, y):
                    buf_deq.appendleft(node)
                    #buf_deq.extend(self.nodes)
                    #self.nodes = buf_deq
                    self.nodes.extend(buf_deq)
                    self.node_clicked_event(node.get_id())
                    self.node_menu_event(node.get_id(),node,e)


                    return
                buf_deq.appendleft(node)
            #self.nodes=buf_deq
            self.nodes.extend(buf_deq)
            print(self.edges)
            buf_deq=deque()
            for ne in range(len(self.edges)):
                edge = self.edges.pop()
                if edge.is_point_belongs_to_edge(x, y):
                    node_a, node_b = edge.get_nodes()
                    buf_deq.appendleft(edge)
                    #buf_deq.extend(self.edges)
                    self.edges.extend(buf_deq)
                    #self.edges=buf_deq
                    self.edge_clicked_event(node_a.get_id(), node_b.get_id())
                    self.edge_menu_event(node_a.get_id(), node_b.get_id(),edge,e)

                    return
                buf_deq.appendleft(edge)
            #self.edges=buf_deq
            self.edges.extend(buf_deq)
            print(self.edges)
            self.background_menu_event(e)
        else:
            pass#self.clear_highlighting()
    def node_menu_event(self,id,node,e):
        self.__service_node_id=id
        node_right_click_menu=QMenu()
        node_right_click_menu.addAction("id: "+str(id)).setEnabled(False)
        if len(str(node.get_text())):
            node_right_click_menu.addSeparator()
            node_right_click_menu.addAction(str(node.get_text())).setEnabled(False)
        node_right_click_menu.addSeparator()
        node_right_click_menu.addAction("Highlight adjacent nodes").triggered.connect(self.highlight_adjacent_nodes)
        node_right_click_menu.addAction("Filter adjacent nodes").triggered.connect(self.filter_adjacent_nodes)
        node_right_click_menu.addAction("Cycle adjacent nodes").triggered.connect(self.cycle_adjacent_nodes)
        node_right_click_menu.addSeparator()
        node_right_click_menu.exec_(e.globalPos())
    def edge_menu_event(self,id1,id2,edge,e):
        #self.__service_egde_id = id
        node_right_click_menu = QMenu()
        """
        
        node_right_click_menu.addSeparator()
        node_right_click_menu.addAction("Node B: " + ).setEnabled(False)
        node_right_click_menu.addSeparator()
        node_right_click_menu.addAction("Value: " + .setEnabled(False)
        node_right_click_menu.addSeparator()
        """
        if self.inc_matrix[self.id_index[id2]][self.id_index[id1]]:
            node_right_click_menu.addAction("◉" + str(id1) + " ↔ ◉" + str(id2)).setEnabled(False)
        else:
            node_right_click_menu.addAction("◉"+str(id1)+" ➞ ◉"+str(id2)).setEnabled(False)
        node_right_click_menu.addAction("↔ = " + str(edge.get_value())).setEnabled(False)
        node_right_click_menu.exec_(e.globalPos())
    def background_menu_event(self,e):
        node_right_click_menu=QMenu()
        node_right_click_menu.addAction("Clear highlighting").triggered.connect(self.clear_highlighting)
        node_right_click_menu.addAction("Clear node filter").triggered.connect(self.clear_node_filter)
        node_right_click_menu.addAction("Set positions as default").triggered.connect(self.set_positions_as_default)
        node_right_click_menu.addSeparator()
        node_right_click_menu.exec_(e.globalPos())


    #_______________________________
    #nodes
    def highlight_adjacent_nodes(self):
        print("n=", self.n)
        id0 = str(self.__service_node_id)
        for nn in range(self.n):
            node = self.nodes.pop()
            id1 = str(node.get_id())

            i1, i2 = self.id_index[id1], self.id_index[id0]
            if self.inc_matrix[i1][i2] or self.inc_matrix[i2][i1] or id0 == id1:
                node.set_selection(True)
            else:
                node.set_selection(False)
            self.nodes.appendleft(node)
        self.repaint()
    def filter_adjacent_nodes(self):
        print("n=",self.n)
        id0 = str(self.__service_node_id)
        for nn in range(self.n):
            node = self.nodes.pop()
            id1=str(node.get_id())

            i1, i2=self.id_index[id1],self.id_index[id0]
            if self.inc_matrix[i1][i2] or self.inc_matrix[i2][i1] or id0==id1:
                node.set_visibility(True)
            else:
                node.set_visibility(False)
            self.nodes.appendleft(node)
        self.repaint()
    def cycle_adjacent_nodes(self):
        id0 = str(self.__service_node_id)
        li=[]
        size_sum=0
        size_max=0
        for nn in range(self.n):
            node = self.nodes.pop()
            id1 = str(node.get_id())
            i1, i2 = self.id_index[id1], self.id_index[id0]
            if (self.inc_matrix[i1][i2] or self.inc_matrix[i2][i1]) and id0!=id1:
                li.append(node)
                size_sum+=node.get_size()
                size_max=max(size_max,node.get_size())
            if id0==id1:
                node_0=node
            self.nodes.appendleft(node)
        P=size_sum + len(li)*5
        R=P/(2*math.pi)
        R=max(R,size_max)+30
        print(node_0.get_id())
        x0,y0=node_0.get_real_position()
        #node_0.set_position(x0,y0)
        #r0=node_0.get_size()
        #x0 += r0
        #y0 += r0
        #li[0].set_position(int(x0 - v0[0]), int(y0 - v0[1]))
        if len(li)==0:
            return
        ang = math.pi * 2 / len(li)
        for i in range(len(li)):
            k=i+1
            #r1 = li[i].get_size()
            #x1 += r1
            #y1 += r1
            #print(x,y)
            li[i].set_position(x0+(R*math.sin(k*ang)),y0+(R*math.cos(k*ang)))

        self.repaint()
    def clear_node_filter(self):
        self.show_all_nodes()
    def set_positions_as_default(self):
        self.position_render()
        self.repaint()
    def clear_highlighting(self):
        for nn in range(self.n):
            node = self.nodes.pop()
            node.set_selection(False)
            self.nodes.appendleft(node)
        self.repaint()
    # @protected
    def mouseMoveEvent(self, e):
        x = e.x()
        y = e.y()
        btns = e.buttons()
        if btns == Qt.LeftButton:
            if not self.event_flag[0]:
                for nn in range(self.n):
                    node = self.nodes.pop()
                    if node.is_point_belongs_to_node(x, y):
                        self.event_flag = [True, node, x, y]
                        self.nodes.appendleft(node)
                        return
                    self.nodes.appendleft(node)
                if not self.event_flag[0]:
                    self.event_flag = [True, -1, x, y]
            else:
                if self.event_flag[1] == -1:
                    for nn in range(self.n):
                        node = self.nodes.pop()
                        node.add_bias(x - self.event_flag[2], y - self.event_flag[3])
                        self.nodes.appendleft(node)
                    self.event_flag[2], self.event_flag[3] = x, y
                    self.repaint()
                else:
                    # Заменить
                    """
                    for nn in range(self.n):
                        node = self.nodes.pop()
                        if self.id_index[node.get_id()]==self.event_flag[1]:
                            node.add_bias(x - self.event_flag[2] , y - self.event_flag[3])
                        self.nodes.appendleft(node)
                    """
                    node = self.event_flag[1]
                    node.add_bias(x - self.event_flag[2], y - self.event_flag[3])
                    print(x - self.event_flag[2], y - self.event_flag[3])
                    self.event_flag[2], self.event_flag[3] = x, y
                    self.repaint()

    # @protected
    def mouseReleaseEvent(self, e):
        self.event_flag = [False, 0, 0, 0]

    # @protected
    def wheelEvent(self, e):
        self.screen_scale_cntr += (e.angleDelta() / 120).y() / 4
        if self.screen_scale_cntr >= 0:
            screen_scale = (self.screen_scale_cntr + 1)
        else:
            screen_scale = (1 / (abs(self.screen_scale_cntr - 1)))

        for nn in range(self.n):
            node = self.nodes.pop()
            node.set_scaling(screen_scale)
            self.nodes.appendleft(node)
        self.repaint()

    def set_render_func(self, func: Callable):
        self.render_function = func

    @private
    def default_pushed_square_render(self, nodes: deque):
        h = 60
        # i_to_id = list(id_index.keys())
        n = len(nodes)
        x = y = int(n ** 0.5)
        if x * y < n:
            y += 1
        for i in range(y):
            for j in range(x):
                if i * x + j >= n:
                    break
                node = nodes.pop()
                node.set_position(j * h + i * 10, i * h)
                nodes.appendleft(node)
                # pos_array.append((([j * h + i * 10, i * h]), id_index[i_to_id[i * x + j]]))

    @protected
    def position_render(self):
        self.render_function(self.nodes)

    def set_antialising_level(self, n: int):
        self.aa_level = n
        self.repaint()

    # @protected
    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        if self.aa_level == 0:
            pass
        elif self.aa_level == 1:
            qp.setRenderHint(QPainter.Antialiasing)
        elif self.aa_level == 2:
            qp.setRenderHint(QPainter.HighQualityAntialiasing)
        else:
            pass
        self.draw_edges(qp)
        self.draw_edges_texts(qp)
        self.draw_nodes(qp)
        self.draw_nodes_texts(qp)
        #qp.drawEllipse(0, 0, 20, 20)
        qp.end()

    @protected
    def draw_edges(self, qp: QPainter):
        for ne in range(len(self.edges)):
            edge = self.edges.popleft()
            edge.draw_edge(qp)
            self.edges.append(edge)

    @protected
    def draw_nodes(self, qp: QPainter):
        for nn in range(self.n):
            node = self.nodes.popleft()
            node.draw_node(qp)
            self.nodes.append(node)

    @protected
    def draw_edges_texts(self, qp: QPainter):
        for ne in range(len(self.edges)):
            edge = self.edges.pop()
            edge.draw_edge_text(qp)
            self.edges.appendleft(edge)

    @protected
    def draw_nodes_texts(self, qp: QPainter):
        for nn in range(self.n):
            node = self.nodes.pop()
            node.draw_node_text(qp)
            self.nodes.appendleft(node)

    def get_nodes(self) -> list:
        return list(self.nodes)

    def get_edges(self) -> list:
        return list(self.edges)

    def clear_nodes_and_edges(self):
        self.n = 0
        self.matrix = np.array([[]])
        self.inc_matrix = np.array([[]])
        self.id_index = dict()
        self.nodes = deque()
        self.edges = deque()
        self.repaint()

    def clear_edges(self):
        self.matrix = np.zeros(self.n ** 2).reshape(self.n, self.n)
        self.inc_matrix = np.zeros(self.n ** 2).reshape(self.n, self.n)
        self.edges = deque()
        self.repaint()

    def set_nodes_visibility_filter(self, ids: list):
        li = ids
        assert len(li) == self.n
        for nn in range(self.n):
            node = self.nodes.pop()
            node.set_visibility(li[nn])
            self.nodes.appendleft(node)
        self.repaint()

    def show_all_nodes(self):
        for nn in range(self.n):
            node = self.nodes.pop()
            node.set_visibility(True)
            self.nodes.appendleft(node)
        self.repaint()

    def show_node(self, id: str):
        for nn in range(self.n):
            node = self.nodes.pop()
            if node.get_id() == id:
                node.set_visibility(True)
            self.nodes.appendleft(node)
        self.repaint()

    def hide_node(self, id: str):
        for nn in range(self.n):
            node = self.nodes.pop()
            if node.get_id() == id:
                node.set_visibility(False)
            self.nodes.appendleft(node)
        self.repaint()

    def set_edges_visibility_filter(self, id_pairs: list):
        li = id_pairs
        assert len(li) == len(self.edges)
        for ne in range(len(self.edges)):
            edge = self.edges.pop()
            edge.set_visibility(li[ne])
            self.nodes.appendleft(edge)
        self.repaint()

    def show_all_edges(self):
        for ne in range(len(self.edges)):
            edge = self.edges.pop()
            edge.set_visibility(True)
            self.edges.appendleft(edge)
        self.repaint()

    def show_edge(self, ida: str, idb: str):
        for ne in range(len(self.edges)):
            edge = self.edges.pop()
            node_a, node_b = edge.get_nodes()
            if ida == node_a.get_id and idb == node_b.get_id():
                edge.set_visibility(True)
            self.edge.appendleft(edge)
        self.repaint()

    def hide_edge(self, ida: str, idb: str):
        for ne in range(len(self.edges)):
            edge = self.edges.pop()
            node_a, node_b = edge.get_nodes()
            if ida == node_a.get_id and idb == node_b.get_id():
                edge.set_visibility(False)
            self.edge.appendleft(edge)
        self.repaint()
