"""
Main Qt widget for the Thorvald II hardware monitor.

Three tabs:
  1. Motor Controllers  — one row per drive, shows mode / volts / amps / temp / speed
  2. Batteries          — one row per BMS, shows voltage / current / SoC / SoH / temp
  3. Enclosure          — comm state, relays, digital + analog pins
"""

import threading

from python_qt_binding.QtCore import Qt, QTimer
from python_qt_binding.QtGui import QColor
from python_qt_binding.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QTableWidget, QTableWidgetItem, QLabel, QGroupBox,
    QGridLayout, QSizePolicy,
)

from thorvald_msgs.msg import ControllerArray, BatteryArray, EnclosureState


# Colour helpers
def _item(text, color=None):
    it = QTableWidgetItem(str(text))
    it.setFlags(Qt.ItemIsEnabled)
    if color:
        it.setBackground(color)
    return it


_GREEN  = QColor(144, 238, 144)
_YELLOW = QColor(255, 255, 153)
_RED    = QColor(255, 102, 102)
_GREY   = QColor(200, 200, 200)


class ThorvaldWidget(QWidget):

    MOTOR_HEADERS = ['ID', 'Mode', 'Volts (V)', 'Amps (A)', 'Temp (°C)',
                     'Speed (rad/s)', 'Position', 'Power (%)']
    BATTERY_HEADERS = ['ID', 'Voltage (V)', 'Current (A)', 'SoC (%)',
                       'SoH (%)', 'Temp (°C)', 'Cycles', 'Charge FET', 'Discharge FET']
    ENCLOSURE_COMM_STATES = {-3: 'NEVER ONLINE', -2: 'LAST UNKNOWN',
                             -1: 'LAST ERROR', 0: 'WAS ONLINE', 1: 'IS ONLINE'}

    def __init__(self, node):
        super().__init__()
        self._node = node
        self._lock = threading.Lock()
        self._controller_msg = None
        self._battery_msg    = None
        self._enclosure_msg  = None

        self._build_ui()

        self._ctrl_sub = node.create_subscription(
            ControllerArray, 'motor_controller_data',
            self._on_controllers, 10)
        self._batt_sub = node.create_subscription(
            BatteryArray, 'battery_data',
            self._on_batteries, 10)
        self._enc_sub = node.create_subscription(
            EnclosureState, 'enclosure_state',
            self._on_enclosure, 10)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_ui)
        self._timer.start(500)   # 2 Hz UI refresh

    # ------------------------------------------------------------------ UI build

    def _build_ui(self):
        self.setWindowTitle('Thorvald II Monitor')
        root = QVBoxLayout(self)

        tabs = QTabWidget()
        root.addWidget(tabs)

        # ── Tab 1: Motor Controllers ──────────────────────────────────
        motor_tab = QWidget()
        motor_layout = QVBoxLayout(motor_tab)
        self._motor_table = QTableWidget(0, len(self.MOTOR_HEADERS))
        self._motor_table.setHorizontalHeaderLabels(self.MOTOR_HEADERS)
        self._motor_table.horizontalHeader().setStretchLastSection(True)
        self._motor_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._motor_status_label = QLabel('Waiting for /motor_controller_data …')
        motor_layout.addWidget(self._motor_status_label)
        motor_layout.addWidget(self._motor_table)
        tabs.addTab(motor_tab, 'Motor Controllers')

        # ── Tab 2: Batteries ──────────────────────────────────────────
        batt_tab = QWidget()
        batt_layout = QVBoxLayout(batt_tab)
        self._batt_table = QTableWidget(0, len(self.BATTERY_HEADERS))
        self._batt_table.setHorizontalHeaderLabels(self.BATTERY_HEADERS)
        self._batt_table.horizontalHeader().setStretchLastSection(True)
        self._batt_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._batt_status_label = QLabel('Waiting for /battery_data …')
        batt_layout.addWidget(self._batt_status_label)
        batt_layout.addWidget(self._batt_table)
        tabs.addTab(batt_tab, 'Batteries')

        # ── Tab 3: Enclosure ──────────────────────────────────────────
        enc_tab = QWidget()
        enc_layout = QVBoxLayout(enc_tab)

        # Comm state banner
        self._enc_comm_label = QLabel('Waiting for /enclosure_state …')
        self._enc_comm_label.setAlignment(Qt.AlignCenter)
        font = self._enc_comm_label.font()
        font.setPointSize(12)
        font.setBold(True)
        self._enc_comm_label.setFont(font)
        enc_layout.addWidget(self._enc_comm_label)

        # Relay / safety row
        relay_box = QGroupBox('Relays & Safety')
        relay_grid = QGridLayout(relay_box)
        self._enc_labels = {}
        for col, key in enumerate(['ESD OK', 'Contactors', 'Motor Power', 'RL3', 'RL5']):
            lbl = QLabel('—')
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setAutoFillBackground(True)
            relay_grid.addWidget(QLabel(key), 0, col, Qt.AlignCenter)
            relay_grid.addWidget(lbl, 1, col)
            self._enc_labels[key] = lbl
        enc_layout.addWidget(relay_box)

        # Digital pins table
        dig_box = QGroupBox('Digital Pins')
        dig_v = QVBoxLayout(dig_box)
        self._dig_table = QTableWidget(0, 2)
        self._dig_table.setHorizontalHeaderLabels(['Pin', 'Value'])
        self._dig_table.setEditTriggers(QTableWidget.NoEditTriggers)
        dig_v.addWidget(self._dig_table)
        enc_layout.addWidget(dig_box)

        # Analog pins table
        ana_box = QGroupBox('Analog Pins')
        ana_v = QVBoxLayout(ana_box)
        self._ana_table = QTableWidget(0, 2)
        self._ana_table.setHorizontalHeaderLabels(['Pin', 'Value'])
        self._ana_table.setEditTriggers(QTableWidget.NoEditTriggers)
        ana_v.addWidget(self._ana_table)
        enc_layout.addWidget(ana_box)

        tabs.addTab(enc_tab, 'Enclosure')

    # ------------------------------------------------------------------ callbacks (ROS thread)

    def _on_controllers(self, msg):
        with self._lock:
            self._controller_msg = msg

    def _on_batteries(self, msg):
        with self._lock:
            self._battery_msg = msg

    def _on_enclosure(self, msg):
        with self._lock:
            self._enclosure_msg = msg

    # ------------------------------------------------------------------ UI refresh (Qt thread)

    def _refresh_ui(self):
        with self._lock:
            ctrl = self._controller_msg
            batt = self._battery_msg
            enc  = self._enclosure_msg

        if ctrl is not None:
            self._update_motors(ctrl)
        if batt is not None:
            self._update_batteries(batt)
        if enc is not None:
            self._update_enclosure(enc)

    def _update_motors(self, msg):
        data = msg.controller_data
        self._motor_status_label.setText(
            f'Last update: {msg.header.stamp.sec}s  |  {len(data)} controller(s)')
        self._motor_table.setRowCount(len(data))

        for row, cd in enumerate(data):
            cs = cd.controller_state
            temp = cs.max_temp

            color = _GREEN if cs.controller_mode == 1 else _YELLOW

            motors = cd.motor_state
            speed_str    = ', '.join(f'{m.speed:.2f}'         for m in motors)
            pos_str      = ', '.join(f'{m.position:.2f}'      for m in motors)
            amps_str     = ', '.join(f'{m.motor_amps:.2f}'    for m in motors)
            power_str    = ', '.join(f'{m.relative_power}'    for m in motors)

            self._motor_table.setItem(row, 0, _item(cd.controller_config.can_id))
            self._motor_table.setItem(row, 1, _item(cs.controller_mode, color))
            self._motor_table.setItem(row, 2, _item(f'{cs.battery_volts:.1f}'))
            self._motor_table.setItem(row, 3, _item(f'{cs.battery_amps:.1f}'))
            temp_color = _RED if temp > 60 else (_YELLOW if temp > 45 else None)
            self._motor_table.setItem(row, 4, _item(temp, temp_color))
            self._motor_table.setItem(row, 5, _item(speed_str))
            self._motor_table.setItem(row, 6, _item(pos_str))
            self._motor_table.setItem(row, 7, _item(power_str))

        self._motor_table.resizeColumnsToContents()

    def _update_batteries(self, msg):
        data = msg.battery_data
        self._batt_status_label.setText(
            f'Last update: {msg.header.stamp.sec}s  |  {len(data)} battery(s)')
        self._batt_table.setRowCount(len(data))

        for row, bd in enumerate(data):
            soc = bd.state_of_charge
            soc_color = _GREEN if soc >= 50 else (_YELLOW if soc >= 20 else _RED)
            temps = bd.temperatures
            temp_str = ', '.join(f'{t:.1f}' for t in temps) if temps else '—'

            self._batt_table.setItem(row, 0, _item(bd.can_id))
            self._batt_table.setItem(row, 1, _item(f'{bd.battery_voltage:.2f}'))
            self._batt_table.setItem(row, 2, _item(f'{bd.battery_current:.2f}'))
            self._batt_table.setItem(row, 3, _item(soc, soc_color))
            self._batt_table.setItem(row, 4, _item(bd.state_of_health))
            self._batt_table.setItem(row, 5, _item(temp_str))
            self._batt_table.setItem(row, 6, _item(bd.cycle_count))
            self._batt_table.setItem(row, 7, _item('ON' if bd.charge_fet_on    else 'OFF'))
            self._batt_table.setItem(row, 8, _item('ON' if bd.discharge_fet_on else 'OFF'))

        self._batt_table.resizeColumnsToContents()

    def _update_enclosure(self, msg):
        state = msg.communication_state
        state_str = self.ENCLOSURE_COMM_STATES.get(state, str(state))
        self._enc_comm_label.setText(f'Comm: {state_str}')

        comm_color = _GREEN if state == 1 else (_YELLOW if state == 0 else _RED)
        pal = self._enc_comm_label.palette()
        pal.setColor(self._enc_comm_label.backgroundRole(), comm_color)
        self._enc_comm_label.setPalette(pal)
        self._enc_comm_label.setAutoFillBackground(True)

        def _relay_color(ok):
            lbl = self._enc_labels.get(ok[0])
            if lbl:
                lbl.setText('YES' if ok[1] else 'NO')
                c = _GREEN if ok[1] else _RED
                p = lbl.palette()
                p.setColor(lbl.backgroundRole(), c)
                lbl.setPalette(p)

        _relay_color(('ESD OK',       msg.esd_ok))
        _relay_color(('Contactors',   msg.contactors_on))
        _relay_color(('Motor Power',  msg.motor_power_control_on))
        _relay_color(('RL3',          msg.rl3_on))
        _relay_color(('RL5',          msg.rl5_on))

        # Digital pins
        pins = list(zip(msg.digital_pin_numbers, msg.digital_pin_values))
        self._dig_table.setRowCount(len(pins))
        for r, (pin, val) in enumerate(pins):
            self._dig_table.setItem(r, 0, _item(pin))
            self._dig_table.setItem(r, 1, _item('HIGH' if val else 'LOW',
                                                 _GREEN if val else _GREY))

        # Analog pins
        apins = list(zip(msg.analog_pin_numbers, msg.analog_pin_values))
        self._ana_table.setRowCount(len(apins))
        for r, (pin, val) in enumerate(apins):
            self._ana_table.setItem(r, 0, _item(pin))
            self._ana_table.setItem(r, 1, _item(val))

    # ------------------------------------------------------------------ cleanup

    def shutdown(self):
        self._timer.stop()
        self._node.destroy_subscription(self._ctrl_sub)
        self._node.destroy_subscription(self._batt_sub)
        self._node.destroy_subscription(self._enc_sub)
