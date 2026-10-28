"""
row_state.py

Wire format for the perception -> control topic. A std_msgs/Float32MultiArray
keeps the stack free of a custom message package; the field order lives only
here, so publisher and subscriber cannot drift apart.
"""

FIELDS = ('lateral_offset', 'heading_error', 'confidence', 'u_near', 'v_near', 'u_far', 'v_far')
TOPIC = '/perception/row_state'


def encode(msg_cls, **values):
    msg = msg_cls()
    msg.data = [float(values.get(name, 0.0)) for name in FIELDS]
    return msg


def decode(msg):
    return dict(zip(FIELDS, msg.data))
