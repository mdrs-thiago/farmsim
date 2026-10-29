"""
sim.py

Starting and stopping the Gazebo + ROS 2 simulation from Python (replaces
sim_ctl.sh). Everything farm_sim.launch.py can spawn shares one GZ_PARTITION,
so a second instance, or an orphan from a killed run, mixes its /odom and
camera frames into the new run. `stop()` therefore kills every matching
process by PID and verifies none is left, and `launch()` refuses to start
while anything is still running.
"""

import os
import signal
import subprocess
import time


GZ_PARTITION = 'farmsim'
ROS_SETUP = os.environ.get('FARMSIM_ROS_SETUP', '/opt/ros/lyrical/setup.bash')

# record_camera_poses is listed although no launch file starts it: an orphaned
# recorder keeps subscribing to the camera and halves the real-time factor of
# every later session.
PATTERN = ('gz-sim-main|gz-sim-gui-client|ros2 launch farmsim_|parameter_bridge .*camera_info|'
           'thorvald_base/twist_to_joints|thorvald_gz/joint_command_splitter|'
           'farmsim_control/controller_node|farmsim_perception/perception_node|'
           'farmsim_perception/record_camera_poses|farmsim_perception/live_|'
           'farmsim_farm/capture_topdown|robot_state_publisher --ros-args|'
           'farmsim_utils.video record')


def tracked_pids():
    out = subprocess.run(['pgrep', '-f', PATTERN], capture_output=True, text=True).stdout
    own = {os.getpid(), os.getppid()}
    return [int(p) for p in out.split() if int(p) not in own]


def stop(verbose=True):
    pids = tracked_pids()
    if not pids:
        return True
    if verbose:
        print(f'[sim] stopping {len(pids)} process(es): {pids}')
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(2)
    left = tracked_pids()
    if left and verbose:
        print(f'[sim] WARNING: still running after SIGKILL: {left}')
    return not left


def ros_env():
    """Environment with ROS + the colcon workspace sourced and the farm models
    on GZ_SIM_RESOURCE_PATH."""
    from farmsim_utils.paths import REPO
    install = REPO / 'install' / 'setup.bash'
    cmd = f'source {ROS_SETUP} >/dev/null && source {install} >/dev/null && env -0'
    raw = subprocess.run(['bash', '-c', cmd], capture_output=True, check=True).stdout
    env = dict(item.split('=', 1) for item in raw.decode().split('\0') if '=' in item)
    from farmsim_farm.fields import model_resource_path
    env['GZ_SIM_RESOURCE_PATH'] = os.pathsep.join(
        [model_resource_path(), env.get('GZ_SIM_RESOURCE_PATH', '')]).strip(os.pathsep)
    env['GZ_PARTITION'] = GZ_PARTITION
    env['FARMSIM_ROOT'] = str(REPO)
    return env


def launch_args(mapping):
    out = []
    for key, value in mapping.items():
        if value is None or value == '':
            continue  # ros2 launch rejects `key:=`; the launch file default applies
        if isinstance(value, bool):
            value = 'true' if value else 'false'
        out.append(f'{key}:={value}')
    return out


class Sim:
    """`with Sim(args, log_path) as sim:` launches farm_sim.launch.py and
    guarantees a clean teardown."""

    def __init__(self, launch_mapping, log_path, launch_file='farm_sim.launch.py',
                 package='farmsim_farm', startup_wait_s=12.0):
        self.cmd = ['ros2', 'launch', package, launch_file] + launch_args(launch_mapping)
        self.log_path = log_path
        self.startup_wait_s = startup_wait_s
        self.proc = None
        self.env = None

    def __enter__(self):
        if tracked_pids():
            stop()
        if tracked_pids():
            raise RuntimeError('sim processes still running after stop(); refusing to launch')
        self.env = ros_env()
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._log = open(self.log_path, 'w')
        print(f'[sim] {" ".join(self.cmd)}')
        self.proc = subprocess.Popen(self.cmd, env=self.env, stdout=self._log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(self.startup_wait_s)
        if self.proc.poll() is not None:
            raise RuntimeError(f'sim exited during startup, see {self.log_path}')
        return self

    def run(self, cmd, log_path=None, wait=False):
        """Start another ROS process in the same environment."""
        out = open(log_path, 'w') if log_path else subprocess.DEVNULL
        proc = subprocess.Popen(cmd, env=self.env, stdout=out, stderr=subprocess.STDOUT,
                                start_new_session=True)
        if wait:
            proc.wait()
        return proc

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGINT)
                self.proc.wait(timeout=10)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                pass
        stop()
        self._log.close()
        return False
