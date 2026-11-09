"""GMR G1 CSV -> pinned SuperMimic motion contract, using CPU kinematics only."""
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from g1_supermimic.support.contract import sha256, write_json
from g1_supermimic.support.motion import inspect
from .runtime import activate


def source_joint_names(path):
    """Read the standalone GMR G1 XML without loading its visual meshes.

    Includes are rejected because their expansion can change the qpos order.
    This adapter supports GMR's pelvis-based, 29-hinge-joint G1 only.
    """
    try:
        doc = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f'Invalid source XML: {path}') from exc
    if doc.findall('.//include'):
        raise ValueError('Source XML must be standalone (no includes)')
    world = doc.find('worldbody')
    roots = [] if world is None else world.findall('body')
    if len(roots) != 1 or roots[0].get('name') != 'pelvis':
        raise ValueError('Expected one GMR G1 root body named pelvis')
    root = roots[0]
    free = [node for node in root.iter() if node.tag == 'freejoint'
            or (node.tag == 'joint' and node.get('type') == 'free')]
    if len(free) != 1 or free[0] not in list(root):
        raise ValueError('Expected one floating joint directly on pelvis')
    joints = [node for node in root.iter('joint') if node.get('type') != 'free']
    if any(node.get('type', 'hinge') != 'hinge' for node in joints):
        raise ValueError('Only scalar hinge joints are supported')
    names = [node.get('name') for node in joints]
    if len(names) != 29 or None in names or len(set(names)) != 29:
        raise ValueError('Source must contain 29 uniquely named G1 joints')
    return names


def validate_csv(raw, input_fps, speed):
    """Reject invalid units/layout before the upstream resampler normalizes quats."""
    if not np.isfinite(input_fps) or input_fps <= 0:
        raise ValueError('input_fps must be finite and positive')
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError('speed must be finite and positive')
    effective_fps = input_fps * speed
    if not np.isfinite(effective_fps) or effective_fps <= 0:
        raise ValueError('input_fps * speed must be finite and positive')
    if raw.ndim != 2 or raw.shape[1] != 36 or len(raw) < 3:
        raise ValueError('Expected at least 3 CSV frames with 36 columns: xyz, xyzw, 29 radians')
    if not np.isfinite(raw).all():
        raise ValueError('CSV contains NaN or Inf')
    if not np.allclose(np.linalg.norm(raw[:, 3:7], axis=1), 1., atol=1e-3, rtol=0):
        raise ValueError('CSV root quaternions must be unit xyzw quaternions')
    duration = (len(raw) - 1) / effective_fps
    if not np.isfinite(duration):
        raise ValueError('Motion duration must be finite')
    if duration < 2 / 50:
        raise ValueError('Motion is too short for at least 3 output frames at 50 Hz')


def import_motion(csv_path, source_xml, output, input_fps, speed=1.):
    """Preserve lateral displacement and rebuild all body states on our G1 model.

    A speed below one lengthens the same trajectory; velocities are recomputed.
    No per-frame ground shift, clipping, mirroring, or loop stitching is applied.
    """
    activate()
    from g1_supermimic.core.data.motion_convert import (
        G1_JOINT_NAMES, g1_xml_path, load_g1_model, resample_motion, forward_kinematics,
    )
    source, output = [Path(p).expanduser().resolve() for p in (csv_path, output)]
    xml = Path(source_xml).expanduser().resolve() if source_xml is not None else None
    if output.suffix != '.npz':
        raise ValueError('Output must end in .npz')
    metadata = output.with_suffix('.json')
    if output.exists() or metadata.exists():
        raise FileExistsError(f'Preserve existing imported motion/metadata: {output}')
    # NJU's published custom_dance CSVs explicitly declare this SDK joint order.
    names = source_joint_names(xml) if xml is not None else list(G1_JOINT_NAMES)
    if set(names) != set(G1_JOINT_NAMES):
        raise ValueError('Source XML joint names differ from the installed 29-DoF G1')
    raw = np.loadtxt(source, delimiter=',', ndmin=2)
    validate_csv(raw, input_fps, speed)
    permutation = [names.index(name) for name in G1_JOINT_NAMES]
    raw[:, 7:] = raw[:, 7:][:, permutation]
    model = load_g1_model()
    # Validate every input pose too, so downsampling cannot hide limit violations.
    joint_ids = np.arange(1, model.njnt)
    limited = model.jnt_limited[joint_ids].astype(bool)
    ranges = model.jnt_range[joint_ids]
    excess = np.maximum(ranges[:, 0] - raw[:, 7:], raw[:, 7:] - ranges[:, 1])
    excess[:, ~limited] = 0
    if np.any(excess > .01):
        frame, joint = np.unravel_index(np.argmax(excess), excess.shape)
        raise ValueError(f'Joint limit violation: frame {frame}, {G1_JOINT_NAMES[joint]}, '
                         f'{excess[frame, joint]:.4f} rad outside limit; fix retargeting')
    motion = resample_motion(raw, input_fps * speed, 50.)
    bodies = forward_kinematics(model, motion)
    arrays = dict(
        fps=np.array([50.], dtype=np.float32),
        joint_pos=motion.joint_pos.astype(np.float32),
        joint_vel=motion.joint_vel.astype(np.float32),
        body_pos_w=bodies.pos_w.astype(np.float32),
        body_quat_w=bodies.quat_w.astype(np.float32),
        body_lin_vel_w=bodies.lin_vel_w.astype(np.float32),
        body_ang_vel_w=bodies.ang_vel_w.astype(np.float32),
        joint_names=np.array(G1_JOINT_NAMES), body_names=np.array(bodies.body_names),
        source=np.array(str(source)), input_fps=np.array([input_fps], dtype=np.float32),
        playback_speed=np.array([speed], dtype=np.float32),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.import-', dir=output.parent) as folder:
        temporary = Path(folder) / 'motion.npz'
        np.savez(temporary, **arrays)
        report = inspect(temporary, G1_JOINT_NAMES, bodies.body_names, 50.)
        report.update(
            path=str(output), source=str(source), source_sha256=sha256(source),
            source_xml=str(xml) if xml is not None else None,
            source_xml_sha256=sha256(xml) if xml is not None else None,
            input_joint_order='source_xml' if xml is not None else 'declared_g1_sdk',
            target_xml=str(g1_xml_path()), target_xml_sha256=sha256(g1_xml_path()),
            source_joint_names=names, joint_permutation=permutation,
            input_fps=float(input_fps), playback_speed=float(speed), input_frames=len(raw),
            input_quaternion='xyzw', coordinate_system='Z-up, metres, radians',
            root_displacement_m=(motion.root_pos[-1] - motion.root_pos[0]).tolist(),
            peak_joint_speed_rad_s=float(np.max(np.abs(motion.joint_vel))),
            peak_root_speed_m_s=float(np.max(np.linalg.norm(motion.root_lin_vel_w, axis=1))),
            max_input_joint_limit_excess_rad=float(max(0., excess.max())),
            validation='Kinematics and data contract only; contact feasibility and dynamic tracking untested',
        )
        # Exclusive creation prevents a simultaneous import overwriting an existing clip.
        with output.open('xb') as stream:
            stream.write(temporary.read_bytes())
    write_json(metadata, report)
    return report
