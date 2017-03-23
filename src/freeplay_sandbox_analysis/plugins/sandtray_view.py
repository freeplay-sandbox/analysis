
from PIL import Image

# HACK workaround for upstream pillow issue python-pillow/Pillow#400
import sys
from python_qt_binding import QT_BINDING_MODULES
if (
    not QT_BINDING_MODULES['QtCore'].__name__.startswith('PyQt5') and
    'PyQt5' in sys.modules
):
    sys.modules['PyQt5'] = None
from PIL.ImageQt import ImageQt

from .topic_message_view import TopicMessageView
import image_helper

from .sandtray_item import SandtrayItem

from python_qt_binding.QtCore import Qt
from python_qt_binding.QtGui import QPixmap
from python_qt_binding.QtWidgets import QGraphicsScene, QGraphicsView


class SandtrayView(TopicMessageView):
    name = 'Sandtray'

    def __init__(self, timeline, parent, topic):
        super(SandtrayView, self).__init__(timeline, parent, topic)

        self._items = {}

        self._image = None
        self._image_topic = None
        self._image_stamp = None
        self.quality = Image.NEAREST  # quality hint for scaling

        self._sandtray = SandtrayItem()

        self._sandtray_view = QGraphicsView(parent)
        self._sandtray_view.resizeEvent = self._resizeEvent
        self._scene = QGraphicsScene()
        self._scene.addItem(self._sandtray)

        self._sandtray_view.setScene(self._scene)
        self._sandtray_view.fitInView(self._scene.itemsBoundingRect(), Qt.KeepAspectRatio)

        parent.layout().addWidget(self._sandtray_view)

    # MessageView implementation
    def _resizeEvent(self, event):
        self._sandtray_view.fitInView(self._scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def message_viewed(self, bag, msg_details):
        """
        render the sandtray
        """
        TopicMessageView.message_viewed(self, bag, msg_details)
        topic, msg, t = msg_details[:3]
        if msg:
            for t in msg.transforms:
                if t.header.frame_id == "sandtray":
                    self._items[t.child_frame_id] = t.transform.translation.x, -t.transform.translation.y
            self._sandtray.update(self._items)

    def message_cleared(self):
        TopicMessageView.message_cleared(self)
        self.set_image(None, None, None)

    # End MessageView implementation

    def put_image_into_scene(self):
        if self._image:
            QtImage = ImageQt(self._image)
            pixmap = QPixmap.fromImage(QtImage)
            self._scene.clear()
            self._scene.addPixmap(pixmap)

    def set_image(self, image_msg, image_topic, image_stamp):
        self._image_msg = image_msg
        if image_msg:
            self._image = image_helper.imgmsg_to_pil(image_msg)
        else:
            self._image = None
        self._image_topic = image_topic
        self._image_stamp = image_stamp
        self.put_image_into_scene()
