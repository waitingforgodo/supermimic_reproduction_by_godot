"""Match the simulator's cached reference alignment in imagined body rewards."""
from dataclasses import dataclass
from functools import wraps
import os


@dataclass
class ModelAlignmentCfg:
    cached_body_reference: bool = False


def alignment_manifest(cfg):
    flag = cfg.model_alignment.cached_body_reference
    if type(flag) is not bool:
        raise ValueError('model_alignment.cached_body_reference must be a JSON boolean')
    if not flag:
        return None
    if int(os.environ.get('WORLD_SIZE', '1')) != 1 or cfg.policy_trainer.compile is not None:
        raise ValueError('Cached reference reward alignment requires single-GPU eager imagination')
    return dict(method='cached_body_reference',
                reference_alignment_pose='robot anchor at s_t, matching command-manager cache',
                reward_robot_pose='robot bodies at predicted s_t+1',
                changed_terms=['motion_body_pos', 'motion_body_ori'],
                simulator_and_evaluation_rewards_changed=False,
                actor_observations_changed=False)


def body_rewards(heads, bodies, ref, previous_anchor):
    """Only the reference alignment uses the PRE-transition robot anchor pose."""
    from g1_supermimic.core.envs.mdp.rewards import reanchor_reference, _sq_quat_err
    i, cfg = heads.anchor_index, heads.reward_cfg
    aligned_pos, aligned_quat = reanchor_reference(
        ref.body_pos_w, ref.body_quat_w, ref.body_pos_w[..., i, :],
        ref.body_quat_w[..., i, :], previous_anchor[..., :3], previous_anchor[..., 3:])
    import torch
    position_error = (aligned_pos - bodies['pos']).square().sum(-1).mean(-1)
    rotation_error = _sq_quat_err(aligned_quat, bodies['quat']).mean(-1)
    return (cfg.dt * cfg.w_body_pos * torch.exp(-position_error / cfg.std_body_pos**2),
            cfg.dt * cfg.w_body_ori * torch.exp(-rotation_error / cfg.std_body_ori**2))


class CachedReferenceStep:
    """Reuse FK across consecutive steps, with a fresh cache for every update."""
    def __init__(self, wm, default_joint_pos):
        self.wm, self.q0 = wm, default_joint_pos
        self.reset()

    def reset(self):
        self._previous_state = self._previous_anchor = None

    def __call__(self, state, action, prev_action, xi, ref):
        import torch
        from g1_supermimic.core.imagination.compiled import REWARD_TERM_NAMES
        from g1_supermimic.core.world_model.heads import analytic_observation
        heads = self.wm.heads
        i = heads.anchor_index
        if self._previous_state is state:
            previous_anchor = self._previous_anchor
        else:
            before = heads.robot_bodies(state)
            previous_anchor = torch.cat([before['pos'][..., i, :], before['quat'][..., i, :]], -1)
        next_state = self.wm.step(state, action, xi)
        bodies = heads.robot_bodies(next_state)
        terms = heads.reward(next_state, action, prev_action, ref, bodies)
        position, rotation = body_rewards(heads, bodies, ref, previous_anchor)
        terms['total'] = (terms['total'] + position - terms['motion_body_pos']
                          + rotation - terms['motion_body_ori'])
        terms['motion_body_pos'], terms['motion_body_ori'] = position, rotation
        anchor = torch.cat([bodies['pos'][..., i, :], bodies['quat'][..., i, :]], -1)
        self._previous_state, self._previous_anchor = next_state, anchor
        return (next_state, terms['total'], torch.stack([terms[k] for k in REWARD_TERM_NAMES]),
                self.wm.termination_logits(next_state, ref),
                analytic_observation(next_state, action, xi, self.q0), anchor)


def bind_reward_alignment(trainer):
    if not trainer.cfg.model_alignment.cached_body_reference:
        return
    learner = trainer.policy_trainer
    step = CachedReferenceStep(trainer.wm, trainer.lib.default_joint_pos)
    learner.step_fn = step
    original = learner.update

    @wraps(original)
    def update(*args, **kwargs):
        step.reset()
        try:
            return original(*args, **kwargs)
        finally:
            # Do not retain the previous rollout's autograd graph, including error paths.
            step.reset()
    learner.update = update
