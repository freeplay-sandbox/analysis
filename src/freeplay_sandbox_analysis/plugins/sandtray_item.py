from python_qt_binding.QtCore import qDebug, QPointF, QRectF, Qt, qWarning, Signal
from python_qt_binding.QtGui import QBrush, QCursor, QColor, QFont, \
                                    QFontMetrics, QPen, QPolygonF
from python_qt_binding.QtWidgets import QGraphicsItem


class SandtrayItem(QGraphicsItem):

    length = 0.6 #m
    width = 0.33 #m
    scale = 1000 # px/m

    def __init__(self):
        super(SandtrayItem, self).__init__()

        self._items = {}

        self._bg_color = QColor(179, 179, 179, 25)
        self._fg_color = QColor(204, 204, 204, 102)
        self._item_color = QColor(204, 0, 0, 255)
        self._cube_color = QColor(204, 100, 0, 255)
        self._border_color = QColor(0, 0, 0, 102)

    # QGraphicsItem implementation
    def boundingRect(self):
        return QRectF(0, 0, SandtrayItem.length * SandtrayItem.scale, SandtrayItem.width * SandtrayItem.scale)

    def paint(self, painter, option, widget):

        painter.fillRect(0, 0, 
                         SandtrayItem.length * SandtrayItem.scale, 
                         SandtrayItem.width * SandtrayItem.scale,
                         painter.background())

        painter.setBrush(QBrush(self._fg_color))
        painter.setPen(QPen(self._border_color, 1))
        painter.drawRect(0, 0, SandtrayItem.length * SandtrayItem.scale, SandtrayItem.width * SandtrayItem.scale)

        for label, pos in self._items.items():
            if "cube" in label:
                painter.setBrush(QBrush(self._cube_color))
                painter.drawRect(pos[0] * SandtrayItem.scale - 5, 
                                 pos[1] * SandtrayItem.scale - 5,
                                 10, 10)
            else:
                painter.setBrush(QBrush(self._item_color))
                painter.drawEllipse(QPointF(pos[0] * SandtrayItem.scale, pos[1] * SandtrayItem.scale), 10, 10)

    def update(self, items):
        self._items = items

        super(SandtrayItem, self).update()
