"""SuperMimic 仿真部署：直接加载训练 checkpoint，在同后端闭环回放。

当前上游没有独立 JIT/ONNX 包导出；这里沿用训练时观测和参考动作接口。
"""


def deploy_sim(args):
    from .inference import play
    return play(args)
