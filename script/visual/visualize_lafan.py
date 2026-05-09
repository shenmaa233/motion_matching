import argparse
import glob
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROBOT_FAKE_HAND_PATH = "data/models/g1/g1_29dof.urdf"
ROBOT_SPHERE_HAND_PATH = "data/models/g1/g1_29dof_spherehand.urdf"
DEFAULT_INPUT_DIR = "data/locomotion"
DEFAULT_TERRAIN_MODEL_DIR = "data/models/terrain"


def _rotate_xy(points_xy: np.ndarray, angle: float) -> np.ndarray:
    cos_angle = float(np.cos(angle))
    sin_angle = float(np.sin(angle))
    rotation = np.asarray(
        (
            (cos_angle, -sin_angle),
            (sin_angle, cos_angle),
        ),
        dtype=np.float64,
    )
    return np.asarray(points_xy, dtype=np.float64) @ rotation.T


def _parse_origin_xyz_rpy(origin: ET.Element | None) -> tuple[np.ndarray, float]:
    if origin is None:
        return np.zeros(3, dtype=np.float64), 0.0

    xyz_tokens = origin.attrib.get("xyz", "0 0 0").split()
    rpy_tokens = origin.attrib.get("rpy", "0 0 0").split()
    xyz = np.asarray([float(token) for token in xyz_tokens], dtype=np.float64)
    yaw = float(rpy_tokens[2]) if len(rpy_tokens) >= 3 else 0.0
    return xyz, yaw


def create_plant(robot_model_path: str, object_model_path: str | list[str] | None = None):
    from pydrake.all import (
        AddMultibodyPlantSceneGraph,
        DiagramBuilder,
        MeshcatVisualizer,
        MultibodyPlant,
        Parser,
        RigidTransform,
        StartMeshcat,
    )

    builder = DiagramBuilder()
    plant = MultibodyPlant(1e-3)
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, plant=plant)
    parser = Parser(plant=plant, scene_graph=scene_graph)
    parser.AddModels(robot_model_path)
    if object_model_path is not None:
        if isinstance(object_model_path, list):
            parser.SetAutoRenaming(True)
            for model_path in object_model_path:
                parser.AddModels(model_path)
        else:
            parser.AddModels(object_model_path)
    parser.AddModels("data/models/ground_box.sdf")
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("ground_link"),
        RigidTransform(np.array([0, 0, -0.5])),
    )
    plant.Finalize()
    meshcat = StartMeshcat()
    meshcat_url = meshcat.web_url() if hasattr(meshcat, "web_url") else None
    if meshcat_url:
        print(f"Meshcat URL: {meshcat_url}")
    vis = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    return plant, vis, diagram


def draw_q_knots(vis, plant, diagram, q_knots, dt):
    vis.DeleteRecording()
    vis.StartRecording()
    t_knots = np.arange(len(q_knots)) * dt
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    vis_context = vis.GetMyMutableContextFromRoot(context)
    for t, q in zip(t_knots, q_knots):
        context.SetTime(t)
        plant.SetPositions(plant_context, q)
        vis.ForcedPublish(vis_context)
    vis.StopRecording()
    vis.PublishRecording()


def step_q_knots(vis, plant, diagram, q_knots, dt, start_frame: int = 0):
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    vis_context = vis.GetMyMutableContextFromRoot(context)
    frame_index = int(np.clip(start_frame, 0, max(len(q_knots) - 1, 0)))

    while True:
        context.SetTime(frame_index * dt)
        plant.SetPositions(plant_context, q_knots[frame_index])
        vis.ForcedPublish(vis_context)
        command = input(
            f"frame {frame_index}/{len(q_knots) - 1}, "
            f"t={frame_index * dt:.3f}s | Enter=next, b=prev, number=jump, q=quit: "
        ).strip()
        if command.lower() in {"q", "quit", "exit"}:
            break
        if command.lower() in {"b", "back", "prev"}:
            frame_index = max(0, frame_index - 1)
            continue
        if command:
            try:
                frame_index = int(np.clip(int(command), 0, len(q_knots) - 1))
            except ValueError:
                print(f"Unknown command: {command}")
            continue
        frame_index = min(len(q_knots) - 1, frame_index + 1)


def _natural_sort_key(text: str):
    parts = re.findall(r"\d+|\D+", text)
    key = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return tuple(key)


def find_files(base_path: str, filter: str = "", extension: str = ".npz"):
    if os.path.isfile(base_path):
        return [base_path] if base_path.endswith(extension) else []
    pattern = os.path.join(base_path, f"*{filter}*{extension}")
    files = glob.glob(pattern)
    return sorted(files, key=lambda p: _natural_sort_key(os.path.basename(p)))


def _load_array(
    data: np.lib.npyio.NpzFile,
    name: str,
    candidate_keys: tuple[str, ...],
) -> tuple[str, np.ndarray]:
    for key in candidate_keys:
        if key in data:
            return key, np.asarray(data[key], dtype=np.float64)

    available_keys = ", ".join(sorted(data.files))
    raise KeyError(
        f"Missing {name}; tried keys {candidate_keys}. "
        f"Available keys: {available_keys}"
    )


def _resolve_root_index(data: np.lib.npyio.NpzFile) -> int:
    if "body_names" not in data:
        return 0

    body_names = [str(name) for name in np.asarray(data["body_names"]).tolist()]
    for candidate in ("pelvis", "base", "root", "torso"):
        if candidate in body_names:
            return body_names.index(candidate)
    return 0


def extract_q_knots(data: np.lib.npyio.NpzFile) -> np.ndarray:
    _, joint_positions = _load_array(
        data,
        "joint positions",
        ("dof_positions", "joint_pos"),
    )
    _, body_positions = _load_array(
        data,
        "body positions",
        ("body_positions", "body_pos_w"),
    )
    _, body_rotations = _load_array(
        data,
        "body rotations",
        ("body_rotations", "body_quat_w"),
    )

    root_index = _resolve_root_index(data)

    root_quaternions = body_rotations[:, root_index, :]
    root_positions = body_positions[:, root_index, :]
    return np.concatenate((root_quaternions, root_positions, joint_positions), axis=1)


def _resolve_terrain_model_path(file_name: str, terrain_model_dir: str) -> str | None:
    """从文件名中解析 terrain 文件夹名和 z_scale，返回对应的 URDF 路径。

    文件名约定：前8个字符为 terrain 文件夹名（如 climb_00），
    后缀包含 z_scale 信息（如 _z_scale_1.0）。
    删除file_name最后的_0000
    """
    terrain_folder = file_name[:8]
    if "z_scale" in file_name:
        idx = file_name.index("z_scale") - 1
        z_scale = file_name[idx:]
        z_scale = re.sub(r"_\d+$", "", z_scale)  # 删除末尾的数字（如 _0000）
    else:
        z_scale = "_z_scale_1.0"
    return os.path.join(terrain_model_dir, terrain_folder, f"multi_boxes{z_scale}.urdf")


def _load_generated_trajectory_index(base_path: str) -> dict[str, dict]:
    base = Path(base_path)
    manifest_path = base / "batch_manifest.json" if base.is_dir() else base.parent / "batch_manifest.json"
    if not manifest_path.exists():
        return {}

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    index: dict[str, dict] = {}
    for trajectory in payload.get("trajectories", []):
        trajectory_name = trajectory.get("trajectory_name")
        trajectory_path = trajectory.get("trajectory_path")
        if trajectory_name:
            index[str(trajectory_name)] = trajectory
        if trajectory_path:
            index[Path(str(trajectory_path)).stem] = trajectory
    return index


def _write_transformed_terrain_urdf(terrain_urdf_path: str, terrain_world_pose: dict | None) -> str:
    if not terrain_world_pose:
        return terrain_urdf_path

    source_path = Path(terrain_urdf_path).resolve()
    root = ET.parse(source_path).getroot()
    global_translation = np.asarray(terrain_world_pose.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    global_yaw = float(np.deg2rad(float(terrain_world_pose.get("yaw_deg", 0.0))))

    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if filename and not os.path.isabs(filename):
            mesh.attrib["filename"] = str((source_path.parent / filename).resolve())

    for joint in root.findall(".//joint"):
        if joint.attrib.get("type") != "fixed":
            continue
        parent = joint.find("parent")
        if parent is None or parent.attrib.get("link") != "world":
            continue
        origin = joint.find("origin")
        if origin is None:
            origin = ET.SubElement(joint, "origin")
        local_translation, local_yaw = _parse_origin_xyz_rpy(origin)
        rotated_local_translation = local_translation.copy()
        rotated_local_translation[:2] = _rotate_xy(local_translation[:2], global_yaw)
        composed_translation = global_translation + rotated_local_translation
        composed_yaw = global_yaw + local_yaw
        origin.attrib["xyz"] = " ".join(str(float(value)) for value in composed_translation.tolist())
        origin.attrib["rpy"] = f"0 0 {composed_yaw}"

    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False, encoding="utf-8") as handle:
        handle.write(ET.tostring(root, encoding="unicode"))
        return handle.name


def _apply_visual_recenter(
    q_knots: np.ndarray,
    terrain_world_pose: dict | None,
) -> tuple[np.ndarray, dict | None]:
    recentered_q_knots = np.asarray(q_knots, dtype=np.float64).copy()
    if recentered_q_knots.shape[0] == 0:
        return recentered_q_knots, terrain_world_pose

    if terrain_world_pose is not None:
        terrain_translation = np.asarray(
            terrain_world_pose.get("translation", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        recentered_q_knots[:, 4:7] -= terrain_translation[None, :]

        recentered_pose = dict(terrain_world_pose)
        recentered_pose["translation"] = [0.0, 0.0, 0.0]
        return recentered_q_knots, recentered_pose

    xy_offset = recentered_q_knots[0, 4:6].copy()
    recentered_q_knots[:, 4] -= xy_offset[0]
    recentered_q_knots[:, 5] -= xy_offset[1]
    return recentered_q_knots, None


def visualize_lafan(
    base_path: str = DEFAULT_INPUT_DIR,
    filter: str = "",
    terrain: bool = False,
    terrain_model_dir: str = DEFAULT_TERRAIN_MODEL_DIR,
    recenter: bool = True,
    step: bool = False,
    start_frame: int = 0,
):
    lafan_files = find_files(base_path, filter=filter)
    generated_trajectory_index = _load_generated_trajectory_index(base_path)
    if terrain:
        plant = vis = diagram = None
        for lafan_file in lafan_files:
            file_name = str(Path(lafan_file).stem)
            print(file_name)
            trajectory_manifest = generated_trajectory_index.get(file_name, {})
            terrain_model_path = trajectory_manifest.get("terrain_path")
            if terrain_model_path is not None:
                terrain_model_path = str(Path(terrain_model_path))
            else:
                terrain_model_path = _resolve_terrain_model_path(file_name, terrain_model_dir)
            terrain_world_pose = trajectory_manifest.get("terrain_world_pose")
            data = np.load(lafan_file, allow_pickle=True)
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            q_knots = extract_q_knots(data)
            if recenter:
                q_knots, terrain_world_pose = _apply_visual_recenter(q_knots, terrain_world_pose)
            if terrain_model_path is not None:
                terrain_model_path = _write_transformed_terrain_urdf(terrain_model_path, terrain_world_pose)
            plant, vis, diagram = create_plant(ROBOT_SPHERE_HAND_PATH, terrain_model_path)
            if step:
                step_q_knots(vis, plant, diagram, q_knots, 1.0 / fps, start_frame=start_frame)
            else:
                draw_q_knots(vis, plant, diagram, q_knots, 1.0 / fps)
            input()
    else:
        plant, vis, diagram = create_plant(ROBOT_FAKE_HAND_PATH)
        for lafan_file in lafan_files:
            print(str(Path(lafan_file).stem))
            data = np.load(lafan_file, allow_pickle=True)
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            q_knots = extract_q_knots(data)
            if recenter:
                q_knots, _ = _apply_visual_recenter(q_knots, None)
            if step:
                step_q_knots(vis, plant, diagram, q_knots, 1.0 / fps, start_frame=start_frame)
            else:
                draw_q_knots(vis, plant, diagram, q_knots, 1.0 / fps)
            input()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize BeyondMimic-style G1 motion files in Drake.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="目录或单个 `.npz` 轨迹文件。",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="Only visualize files whose names contain this substring.",
    )
    parser.add_argument(
        "--terrain",
        action="store_true",
        help="同时加载 terrain 模型（从文件名解析 terrain 文件夹和 z_scale）。",
    )
    parser.add_argument(
        "--terrain-model-dir",
        type=str,
        default=DEFAULT_TERRAIN_MODEL_DIR,
        help="terrain URDF 模型的根目录。",
    )
    parser.add_argument(
        "--no-recenter",
        action="store_true",
        help="关闭可视化归一化；默认会把 terrain 平移到世界中心附近，若没有 terrain 则回到第一帧 root 附近。",
    )
    parser.add_argument(
        "--step",
        action="store_true",
        help="逐帧查看并在终端显示 frame index。",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="逐帧模式的初始帧号。",
    )
    args = parser.parse_args()
    visualize_lafan(
        base_path=args.input_dir,
        filter=args.filter,
        terrain=args.terrain,
        terrain_model_dir=args.terrain_model_dir,
        recenter=not args.no_recenter,
        step=args.step,
        start_frame=args.start_frame,
    )
