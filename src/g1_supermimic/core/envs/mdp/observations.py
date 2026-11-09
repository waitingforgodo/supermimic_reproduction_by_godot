# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yuxi Liu; Godot adaptations. See THIRD_PARTY_NOTICES.md.
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from g1_supermimic.core.envs.mdp.commands import MultiClipMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def goal_future_frames(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """(N, K*68) future reference-frame features (plan D5)."""
    command = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return command.goal
