from rqt_gui_py.plugin import Plugin
from .thorvald_widget import ThorvaldWidget


class ThorvaldGUIPlugin(Plugin):

    def __init__(self, context):
        super().__init__(context)
        self.setObjectName('ThorvaldGUIPlugin')
        self._widget = ThorvaldWidget(context.node)
        context.add_widget(self._widget)

    def shutdown_plugin(self):
        self._widget.shutdown()

    def save_settings(self, plugin_settings, instance_settings):
        pass

    def restore_settings(self, plugin_settings, instance_settings):
        pass
