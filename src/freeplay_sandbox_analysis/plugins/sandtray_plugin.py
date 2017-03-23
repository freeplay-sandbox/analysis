
from .plugin import Plugin

from .sandtray_view import SandtrayView


class SandtrayPlugin(Plugin):

    def __init__(self):
        pass

    def get_view_class(self):
        return SandtrayView

    def get_renderer_class(self):
        return None

    def get_message_types(self):
        return ['tf2_msgs/TFMessage', 'tf_msgs/TFMessage']
