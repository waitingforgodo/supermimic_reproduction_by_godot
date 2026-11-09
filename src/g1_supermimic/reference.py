"""回放专用的同步参考影子：只写可视化 qpos，不改变训练/仿真状态。"""
import copy
import math
from types import MethodType

import numpy as np


class ReferenceGhost:
    """使用动作库当前帧及相同世界原点，可选偏移便于并排比较。"""
    def __init__(self, command, enabled=True, alpha=.35, offset=(0., 0., 0.)):
        if not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError('--reference-alpha must be finite and in (0, 1]')
        if len(offset) != 3 or not all(math.isfinite(value) for value in offset):
            raise ValueError('--reference-offset requires three finite meter values')
        self.command, self.enabled, self.alpha = command, enabled, alpha
        self.offset = np.asarray(offset, dtype=float)
        self.model = None

    def draw(self, visualizer):
        """同帧参考根姿态/关节角与策略共享相机；不把根部对齐掩盖轨迹漂移。"""
        if not self.enabled:
            return
        command = self.command
        indices = visualizer.get_env_indices(command.num_envs)
        if not indices:
            return
        if self.model is None:
            self.model = copy.deepcopy(command._env.sim.mj_model)
            for geom in range(self.model.ngeom):
                if self.model.geom_contype[geom] or self.model.geom_conaffinity[geom]:
                    self.model.geom_rgba[geom, 3] = 0.
                else:
                    self.model.geom_rgba[geom] = (.2, .6, 1., self.alpha)
        visible = self.model.geom_rgba[:, 3] > 0
        self.model.geom_rgba[visible, 3] = self.alpha  # Native viewer 使用模型 alpha。
        indexing = command.robot.indexing
        free_q = indexing.free_joint_q_adr.cpu().numpy()
        joint_q = indexing.joint_q_adr.cpu().numpy()
        # 只传输正在显示的环境，避免为不可见环境做 GPU -> CPU 同步。
        root = command.root_pos_w[indices].detach().cpu().numpy()
        quat = command.root_quat_w[indices].detach().cpu().numpy()
        joints = command.joint_pos[indices].detach().cpu().numpy()
        for row, env_index in enumerate(indices):
            qpos = self.model.qpos0.copy()
            qpos[free_q[:3]] = root[row] + self.offset
            qpos[free_q[3:7]] = quat[row]
            qpos[joint_q] = joints[row]
            visualizer.add_ghost_mesh(qpos, model=self.model, alpha=self.alpha,
                                      label=f'training_reference_{env_index}')

    def create_gui(self, server, on_change=None):
        """网页里直接开关影子、调整透明度和横向偏移。"""
        with server.gui.add_folder('Reference motion / 训练参考动作'):
            enabled = server.gui.add_checkbox('Blue reference ghost', initial_value=self.enabled)
            alpha = server.gui.add_slider('Reference opacity', min=.05, max=1., step=.05, initial_value=self.alpha)
            offset = server.gui.add_vector3('Reference offset (m)', initial_value=tuple(self.offset), step=.1)
            def update(event):
                self.enabled, self.alpha = enabled.value, alpha.value
                self.offset = np.asarray(offset.value, dtype=float)
                if on_change:
                    on_change()
            for control in (enabled, alpha, offset):
                control.on_update(update)


def install_reference_ghost(env, **settings):
    """替换本回放实例的绘制回调，Native/Viser 共用同一参考帧。"""
    command = env.command_manager.get_term('motion')
    ghost = ReferenceGhost(command, **settings)
    command.cfg.debug_vis = True
    command._debug_vis_impl = MethodType(lambda self, visualizer: ghost.draw(visualizer), command)
    original_gui = command.create_gui
    def create_gui(self, name, server, get_env_idx, on_change=None, request_action=None):
        original_gui(name, server, get_env_idx, on_change=on_change, request_action=request_action)
        ghost.create_gui(server, on_change)
    command.create_gui = MethodType(create_gui, command)
    return ghost
