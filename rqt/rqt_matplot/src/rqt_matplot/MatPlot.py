#!/usr/bin/env python

# Copyright (c) 2011, Dorian Scholz, TU Darmstadt
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   * Neither the name of the TU Darmstadt nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import os

from qt_gui.QtBindingHelper import loadUi
from QtCore import qDebug, Qt, QTimer, Slot
from QtGui import QWidget

import roslib
roslib.load_manifest('rqt_matplot')
import rospy
from rxtools.rosplot import ROSData
from rostopic import get_topic_type

import rqt_matplot.MatDataPlot
import rqt_py_common.TopicCompleter

# main class inherits from the ui window class
class MatPlot(QWidget):

    def __init__(self, context):
        super(MatPlot, self).__init__()
        self.setObjectName('MatPlot')

        ui_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'MatPlot.ui')
        loadUi(ui_file, self, {'MatDataPlot': rqt_matplot.MatDataPlot.MatDataPlot})

        if context.serial_number() > 1:
            self.setWindowTitle(self.windowTitle() + (' (%d)' % context.serial_number()))

        self.subscribe_topic_button.setEnabled(False)

        self._topic_completer = rqt_py_common.TopicCompleter.TopicCompleter(self.topic_edit)
        self.topic_edit.setCompleter(self._topic_completer)

        self._start_time = rospy.get_time()
        self._rosdata = {}

        # setup drag 'n drop
        self.data_plot.dropEvent = self.dropEvent
        self.data_plot.dragEnterEvent = self.dragEnterEvent

        # add our self to the main window
        context.add_widget(self)

        # init and start update timer for plot
        self._update_plot_timer = QTimer(self)
        self._update_plot_timer.timeout.connect(self.update_plot)
        self._update_plot_timer.start(40)


    def update_plot(self):
        for topic_name, rosdata in self._rosdata.items():
            data_x, data_y = rosdata.next()
            self.data_plot.update_value(topic_name, data_x, data_y)
        self.data_plot.draw_plot()


    def _get_field_type(self, topic_name):
        # get message
        topic_type, _, message_evaluator = get_topic_type(topic_name)
        if topic_type is None:
            return None
        message = roslib.message.get_message_class(topic_type)()

        # return field type
        if message_evaluator:
            try:
                field_type = type(message_evaluator(message))
            except Exception:
                field_type = None
        else:
            field_type = type(message)

        return field_type


    @Slot('QDragEnterEvent*')
    def dragEnterEvent(self, event):
        if not event.mimeData().hasText():
            if not hasattr(event.source(), 'selectedItems') or len(event.source().selectedItems()) == 0:
                qDebug('Plot.dragEnterEvent(): not hasattr(event.source(), selectedItems) or len(event.source().selectedItems()) == 0')
                return
            item = event.source().selectedItems()[0]
            ros_topic_name = item.data(0, Qt.UserRole)
            if ros_topic_name == None:
                qDebug('Plot.dragEnterEvent(): not hasattr(item, ros_topic_name_)')
                return

        # get topic name
        if event.mimeData().hasText():
            topic_name = str(event.mimeData().text())
        else:
            droped_item = event.source().selectedItems()[0]
            topic_name = str(droped_item.data(0, Qt.UserRole))

        # check for numeric field type
        field_type = self._get_field_type(topic_name)
        if field_type in (int, float):
            event.acceptProposedAction()
        else:
            qDebug('Plot.dragEnterEvent(): rejecting topic "%s" of non-numeric type "%s"' % (topic_name, field_type))


    @Slot('QDropEvent*')
    def dropEvent(self, event):
        if event.mimeData().hasText():
            topic_name = str(event.mimeData().text())
        else:
            droped_item = event.source().selectedItems()[0]
            topic_name = str(droped_item.data(0, Qt.UserRole))
        self.add_topic(topic_name)


    @Slot(str)
    def on_topic_edit_textChanged(self, topic_name):
        # on empty topic name, update topics 
        if topic_name in ('', '/'):
            self._topic_completer.update_topics()

        # check for numeric field type
        field_type = self._get_field_type(topic_name)
        if field_type in (int, float):
            self.subscribe_topic_button.setEnabled(True)
            self.subscribe_topic_button.setToolTip('topic "%s" is numeric: %s' % (topic_name, field_type))
        else:
            self.subscribe_topic_button.setEnabled(False)
            self.subscribe_topic_button.setToolTip('topic "%s" is NOT numeric: %s' % (topic_name, field_type))


    @Slot()
    def on_subscribe_topic_button_clicked(self):
        self.add_topic(str(self.topic_edit.text()))


    def add_topic(self, topic_name):
        if topic_name in self._rosdata:
            qDebug('Plot.add_topic(): topic already subscribed: %s' % topic_name)
            return

        self._rosdata[topic_name] = ROSData(topic_name, self._start_time)
        data_x, data_y = self._rosdata[topic_name].next()
        self.data_plot.add_curve(topic_name, data_x, data_y)



    @Slot()
    def on_clear_button_clicked(self):
        self.clean_up_subscribers()


    @Slot(bool)
    def on_pause_button_clicked(self, checked):
        if checked:
            self._update_plot_timer.stop()
        else:
            self._update_plot_timer.start(40)


    def clean_up_subscribers(self):
        for topic_name, rosdata in self._rosdata.items():
            rosdata.close()
            self.data_plot.remove_curve(topic_name)
        self._rosdata = {}


    def save_settings(self, global_settings, perspective_settings):
        pass


    def restore_settings(self, global_settings, perspective_settings):
        pass


    def set_name(self, name):
        self.setWindowTitle(name)


    # override Qt's closeEvent() method to trigger plugin unloading
    def closeEvent(self, event):
        event.ignore()
        self.deleteLater()


    def close_plugin(self):
        self.clean_up_subscribers()
        QDockWidget.close(self)