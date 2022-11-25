"""
Records a ShapeStack scenario in mono and stereo images. Loops over all
specified camera names and renders the simulation state in the specified
modalitiy. Also records whether the stack tower collapses after the given
simulation time.

Execute with:
LD_PRELOAD=/usr/lib/nvidia-384/libOpenGL.so python3 record_scenario.py ...
"""

import sys
import os
import argparse
import pickle
import math
import random
import re
import xml.etree.ElementTree as ET
import h5py
import tqdm

import numpy as np
import scipy.misc

import mujoco_py
from mujoco_py.modder import LightModder, CameraModder

sys.path.insert(0, os.environ['SHAPESTACKS_CODE_HOME'])
from utilities.mujoco_utils import mjsim_mat_id2name
from shapestacks_renderer.rendering_constants import OBJ_COLORS_RGBA, VSEG_COLOR_CODES, \
  ISEG_COLOR_CODES
import matplotlib
matplotlib.use('TkAgg')


EPS = np.finfo(float).eps * 4.
# ---------- command line arguments ----------
ARGPARSER = argparse.ArgumentParser(
    description='Record a ShapeStacks scenario as a MuJoCo simulation.')
# model setup and directories
ARGPARSER.add_argument(
    '--mjmodel_path', type=str,
    help="The path to the XML definition of the MuJoCo model.")
ARGPARSER.add_argument(
    '--mjsim_state_path', type=str,
    help="Path to a simulation state snapshot. If empty, simulation will start \
    from scratch.")
ARGPARSER.add_argument(
    '--record_path', type=str,
    help="The target directory to store the recordings in.")
# simulation settings
ARGPARSER.add_argument(
    '--mjsim_time', type=int, default=5,
    help="Simulation time in seconds.")
# recording settings
ARGPARSER.add_argument(
    '--fps', type=int, default=8,
    help="The number of frames to record per second (FPS).")
ARGPARSER.add_argument(
    '--max_frames', type=int, default=1,
    help="The maximum number of frames to record. Simulation still continues \
    until max. time is exceeded.")
ARGPARSER.add_argument(
    '--res', dest='resolution', type=int, nargs=2, default=[224, 224],
    help="The image resolution of the recording (height, width).")
ARGPARSER.add_argument(
    '--cameras', type=str, nargs='+', default=['cam_1'],
    help="The cameras to record with. Listed camera names must match with \
    camera names in the MJCF model definition.")
ARGPARSER.add_argument(
    '--cam_height_offset', type=float, default=0.0,
    help="Adjust all camera heights by the given value.")
# modality setup
ARGPARSER.add_argument(
    # TODO: deprecate list, allow only one! (rendering artifacts on Ubuntu with GPU)
    '--formats', type=str, nargs='+', default=['rgb'],
    help="The formats to record in. Available formats are: rgb | vseg | iseg | depth.")
ARGPARSER.add_argument(
    '--with_stereo', action='store_true',
    help="Also store stereo images of modalities.")
# RGB settings
ARGPARSER.add_argument(
    '--lightid', type=int, default=0,
    help="The ID of the main scene light (which casts shadows). Must match to \
    a light defined as 'light_<id>' in the MJCF model definition.")
ARGPARSER.add_argument(
    '--walltex', type=int, default=0,
    help="The ID of the wall texture. Must be in the range of the 'mat_wall_*' \
    materials defined in the <asset>-tag of the MJCF model definition.")
ARGPARSER.add_argument(
    '--floortex', type=int, default=0,
    help="The ID of the floor texture. Must be in the range of the 'mat_floor_*' \
    materials defined in the <asset>-tag of the MJCF model definition.")
ARGPARSER.add_argument(
    '--color_mode', type=str, default='original',
    help="Color mode for the shapes. Available modes are: original | uniform | random.")
# file settings
ARGPARSER.add_argument(
    '--file_format', type=str, default='png',
    help="The image format to save the file in. Available: png | eps")


# CONSTANTS

# simulator
BURN_IN_STEPS = 50 # 'burn-in' steps for simulation to reach a stable state
VELOCITY_TOLERANCE = 0.2 # velocities below this threshold are considered 'no movement'

def quat2mat(quaternion):
    """
    Converts given quaternion to matrix.
    Args:
        quaternion (np.array): (x,y,z,w) vec4 float angles
    Returns:
        np.array: 3x3 rotation matrix
    """
    # awkward semantics for use with numba
    inds = np.array([3, 0, 1, 2])
    q = np.asarray(quaternion).copy().astype(np.float32)[inds]

    n = np.dot(q, q)
    if n < EPS:
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q2 = np.outer(q, q)
    return np.array(
        [
            [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
            [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
            [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
        ]
    )

def make_pose(translation, rotation):
    """
    Makes a homogeneous pose matrix from a translation vector and a rotation matrix.
    Args:
        translation (np.array): (x,y,z) translation value
        rotation (np.array): a 3x3 matrix representing rotation
    Returns:
        pose (np.array): a 4x4 homogeneous matrix
    """
    pose = np.zeros((4, 4))
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    pose[3, 3] = 1.0
    return pose
# ---------- helper functions ----------

def _get_cam_light_name(camera_name: str) -> str:
  name_parts = camera_name.split('_')
  cam_name = '_'.join(name_parts[:-1])
  cam_num = name_parts[-1]
  return cam_name + "_light_" + cam_num

def _get_main_light_name(lightid: int) -> str:
  return "light_" + str(lightid)


# ---------- scene setup ----------

def _init_scene_rgb(
    sim: mujoco_py.MjSim, world_xml: ET.Element,
    lightid: int, walltex: int, floortex: int,
    color_mode: str) -> mujoco_py.MjSim:
  """
  Initialize the RGB scene.
  """
  # set main light to cast shadow
  lm = mujoco_py.modder.LightModder(sim)
  light_name = _get_main_light_name(lightid)
  # lm.set_castshadow(light_name, 1)

  # find materials
  mat_id2name = mjsim_mat_id2name(sim)
  mat_floor_id2name = dict(
      filter(lambda t: t[1].startswith('mat_floor_'), mat_id2name.items()))
  mat_wall_id2name = dict(
      filter(lambda t: t[1].startswith('mat_wall_'), mat_id2name.items()))
  matid_floortex = sorted(mat_floor_id2name.items(), key=lambda t: t[0])[floortex][0]
  # matid_walltex = sorted(mat_wall_id2name.items(), key=lambda t: t[0])[walltex][0]

  # set wall and floor materials
  for plane, matid in [
      ('floor', matid_floortex)]:
      # ('floor', matid_floortex), ('wall_1', matid_walltex), ('wall_2', matid_walltex)]:
    geom_id = sim.model.geom_name2id(plane)
    sim.model.geom_matid[geom_id] = matid

  # set RGBA of shapes according to color_mode
  if color_mode == 'original':
    pass # keep original colors
  elif color_mode == 'shuffle': # shuffle the object colors, but keep them unique
    random.shuffle(OBJ_COLORS_RGBA)
    for i, mj_geom_name in enumerate(
        filter(
            lambda name: name.startswith('shape_') or name.startswith('obj_'),
            sim.model.geom_names)):
      mj_geom_id = sim.model.geom_name2id(mj_geom_name)
      sim.model.geom_rgba[mj_geom_id] = OBJ_COLORS_RGBA[i % len(OBJ_COLORS_RGBA)]
  elif color_mode == 'random': # assign random object colors
    for i, mj_geom_name in enumerate(
        filter(
            lambda name: name.startswith('shape_') or name.startswith('obj_'),
            sim.model.geom_names)):
      mj_geom_id = sim.model.geom_name2id(mj_geom_name)
      random.shuffle(OBJ_COLORS_RGBA)
      sim.model.geom_rgba[mj_geom_id] = OBJ_COLORS_RGBA[0]
  else:
    raise Exception("Unknown color mode %s!" % color_mode)

  # advance simulation by one step to update rendering
  sim.step()
  return sim

def _init_scene_vseg(
    sim: mujoco_py.MjSim, world_xml: ET.Element,
    height: int, violations: set) -> mujoco_py.MjSim:
  """
  Initialize the scene for violation segmentation rendering.
  """
  pivot_point = height + 1
  for obj_id in range(1, height+1):

    # determine pivot point
    if obj_id in violations:
      pivot_point = obj_id

    # get geom
    mj_geom_name = 'shape_%s' % obj_id
    mj_geom_id = sim.model.geom_name2id(mj_geom_name)

    # determine color
    if obj_id < pivot_point: # stable lower part
      sim.model.geom_rgba[mj_geom_id] = np.array(VSEG_COLOR_CODES[1], dtype=np.float32)
    elif obj_id == pivot_point: # violating object
      sim.model.geom_rgba[mj_geom_id] = np.array(VSEG_COLOR_CODES[2], dtype=np.float32)
    elif obj_id == pivot_point + 1: # object above violation
      sim.model.geom_rgba[mj_geom_id] = np.array(VSEG_COLOR_CODES[3], dtype=np.float32)
    elif obj_id > pivot_point + 1: # unstable upper part
      sim.model.geom_rgba[mj_geom_id] = np.array(VSEG_COLOR_CODES[4], dtype=np.float32)

  # advance simulation by one step to update rendering
  sim.step()
  return sim

def _init_scene_iseg(
    sim: mujoco_py.MjSim, world_xml: ET.Element,
    height: int) -> mujoco_py.MjSim:
  """
  Initialize the scene for instance segmentation rendering.
  """
  for obj_id in range(1, height+1):

    # get geom
    mj_geom_name = 'shape_%s' % obj_id
    mj_geom_id = sim.model.geom_name2id(mj_geom_name)

    # set color
    sim.model.geom_rgba[mj_geom_id] = np.array(ISEG_COLOR_CODES[obj_id], dtype=np.float32)

  # advance simulation by one step to update rendering
  sim.step()
  return sim


# ---------- rendering setup ----------

def _setup_render_rgb(sim: mujoco_py.MjSim) -> mujoco_py.MjSim:
  # create copy of simulation to customize rendering context
  # flags defined in mjvisualize.h
  render_sim = mujoco_py.MjSim(sim.model)
  render_sim.set_state(sim.get_state())
  render_ctx = mujoco_py.MjRenderContextOffscreen(render_sim, device_id=0)
  # render_ctx = mujoco_py.MjViewer(render_sim)
  render_ctx.scn.stereo = 2 # side-by-side rendering
  return render_sim

def _setup_render_seg(sim: mujoco_py.MjSim) -> mujoco_py.MjSim:
  # create copy of simulation to customize rendering context
  # flags defined in mjvisualize.h
  render_sim = mujoco_py.MjSim(sim.model)
  render_sim.set_state(sim.get_state())
  render_ctx = mujoco_py.MjRenderContextOffscreen(render_sim)
  render_ctx.vopt.flags[1] = 0 # textures off
  render_ctx.vopt.flags[17] = 0 # static body off
  render_ctx.scn.flags[0] = 0 # shadow off
  render_ctx.scn.flags[2] = 0 # reflection off
  render_ctx.scn.flags[4] = 0 # skybox off
  render_ctx.scn.stereo = 2 # side-by-side rendering
  return render_sim

def _setup_render_depth(sim: mujoco_py.MjSim) -> mujoco_py.MjSim:
  # create copy of simulation to customize rendering context
  # flags defined in mjvisualize.h
  render_sim = mujoco_py.MjSim(sim.model)
  render_sim.set_state(sim.get_state())
  render_ctx = mujoco_py.MjRenderContextOffscreen(render_sim)
  render_ctx.vopt.flags[1] = 0 # textures off
  render_ctx.scn.flags[0] = 0 # shadow off
  render_ctx.scn.flags[2] = 0 # reflection off
  render_ctx.scn.flags[4] = 0 # skybox off
  render_ctx.scn.stereo = 2 # side-by-side rendering
  return render_sim

def setup_render(sim: mujoco_py.MjSim, modality: str) -> mujoco_py.MjSim:
  if modality == 'rgb':
    render_sim = _setup_render_rgb(sim)
  elif modality == 'vseg' or modality == 'iseg':
    render_sim = _setup_render_seg(sim)
  elif modality == 'depth':
    render_sim = _setup_render_depth(sim)
  else:
    raise NotImplementedError("Rendering of modality %s is not implemented!" % modality)
  return render_sim


# ---------- rendering functions ----------

def _render_rgb(
    sim: mujoco_py.MjSim,
    camera: str, render_height: int, render_width: int,
    world_xml: ET.Element):
  lm = LightModder(sim)
  cam_names = [c.attrib['name'] for c in world_xml.findall(".//camera")]
  # lights off for all other cameras except the recording one
  for cam in cam_names:
    light_name = _get_cam_light_name(cam)
    # if light_name in sim.model.light_names:
    #   lm.set_active(light_name, 1 if cam == camera else 0)
  # take screenshot
  frame = sim.render(render_width * 2, render_height, camera_name=camera)
  # reset camera lights
  for cam in cam_names:
    light_name = _get_cam_light_name(cam)
    if light_name in sim.model.light_names:
      lm.set_active(light_name, 1)
  return frame

def _convert_depth_to_meters(sim, depth):
    extent = sim.model.stat.extent
    near = sim.model.vis.map.znear * extent
    far = sim.model.vis.map.zfar * extent
    image = near / (1 - depth * (1 - near / far))
    return image

def _create_uniform_pixel_coords_image(resolution: np.ndarray):
    pixel_x_coords = np.reshape(
        np.tile(np.arange(resolution[1]), [resolution[0]]),
        (resolution[0], resolution[1], 1)).astype(np.float32)
    pixel_y_coords = np.reshape(
        np.tile(np.arange(resolution[0]), [resolution[1]]),
        (resolution[1], resolution[0], 1)).astype(np.float32)
    pixel_y_coords = np.transpose(pixel_y_coords, (1, 0, 2))
    uniform_pixel_coords = np.concatenate(
        (pixel_x_coords, pixel_y_coords, np.ones_like(pixel_x_coords)), -1)
    return uniform_pixel_coords

def _transform(coords, trans):
    h, w = coords.shape[:2]
    coords = np.reshape(coords, (h * w, -1))
    coords = np.transpose(coords, (1, 0))
    transformed_coords_vector = np.matmul(trans, coords)
    transformed_coords_vector = np.transpose(
        transformed_coords_vector, (1, 0))
    return np.reshape(transformed_coords_vector,
                      (h, w, -1))

def _pixel_to_world_coords(pixel_coords, cam_proj_mat_inv):
    h, w = pixel_coords.shape[:2]
    pixel_coords = np.concatenate(
        [pixel_coords, np.ones((h, w, 1))], -1)
    world_coords = _transform(pixel_coords, cam_proj_mat_inv)
    world_coords_homo = np.concatenate(
        [world_coords, np.ones((h, w, 1))], axis=-1)
    return world_coords_homo

def _convert_rgbd_to_pointcloud(sim: mujoco_py.MjSim,
                                camera: str, render_height: int, render_width: int,
                                world_xml: ET.Element,
                                depth: np.array):
    camera_id = sim.model.camera_name2id(camera)
    fovy = sim.model.cam_fovy[camera_id]
    f = 0.5*render_height / (math.tan(fovy * np.pi/360))
    # intrinsics = np.array(((-f, 0, render_width / 2), (0, f, render_height / 2), (0, 0, 1)))
    intrinsics = np.array(((f, 0, render_width / 2), (0, f, render_height / 2), (0, 0, 1)))

    cm = CameraModder(sim)

    cam_pos = cm.get_pos(camera)
    cam_quat = cm.get_quat(camera)
    # mat = quat2mat(cam_quat)
    mat = quat2mat(np.array([*cam_quat[1:], cam_quat[0]]))
    extrinsics = make_pose(cam_pos, mat)
    # extrinsics = np.concatenate([mat, cam_pos[:, None]], axis=1)
    camera_axis_correction = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    extrinsics = extrinsics @ camera_axis_correction


    # real_depth = _convert_depth_to_meters(sim, np.flip(depth, axis=0))
    real_depth = _convert_depth_to_meters(sim, depth)
    upc = _create_uniform_pixel_coords_image(real_depth.shape)
    pc = upc * np.expand_dims(real_depth, -1)
    C = np.expand_dims(extrinsics[:3, 3], 0).T
    R = extrinsics[:3, :3]
    R_inv = R.T  # inverse of rot matrix is transpose
    R_inv_C = np.matmul(R_inv, C)
    extrinsics = np.concatenate((R_inv, -R_inv_C), -1)
    cam_proj_mat = np.matmul(intrinsics, extrinsics)
    cam_proj_mat_homo = np.concatenate(
        [cam_proj_mat, [np.array([0, 0, 0, 1])]])
    cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]
    world_coords_homo = np.expand_dims(_pixel_to_world_coords(
        pc, cam_proj_mat_inv), 0)
    world_coords = world_coords_homo[..., :-1][0]

    # x = world_coords[:, :, 0]
    # y = world_coords[:, :, 1]
    # z = world_coords[:, :, 2]
    #
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import proj3d
    #
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # ax.scatter(x.reshape(-1), y.reshape(-1), z.reshape(-1))
    # plt.show()

    return world_coords

def _render_seg(sim: mujoco_py.MjSim, camera: str, render_height: int, render_width: int, world_xml: ET.Element):
  lm = LightModder(sim)
  # switch all lights off
  light_names = [l.attrib['name'] for l in world_xml.findall(".//light")]
  # for light_name in light_names:
  #   lm.set_active(light_name, 0)
  # take screenshot
  frame = sim.render(render_width * 2, render_height, camera_name=camera)
  # reset lights
  for light_name in light_names:
    lm.set_active(light_name, 1)
  return frame

def _render_depth(sim: mujoco_py.MjSim, camera: str, render_height: int, render_width: int, world_xml: ET.Element):
  # take screenshot
  frame = sim.render(render_width * 2, render_height, camera_name=camera, depth=True)
  return frame[1] # depth buffer

def render_modality(sim: mujoco_py.MjSim, modality: str, camera: str, render_height: int, render_width: int, world_xml: ET.Element):
  if modality == 'rgb':
    frame = _render_rgb(sim, camera, render_height, render_width, world_xml)
  elif modality == 'vseg' or modality == 'iseg':
    frame = _render_seg(sim, camera, render_height, render_width, world_xml)
  elif modality == 'depth':
    frame = _render_depth(sim, camera, render_height, render_width, world_xml)
  else:
    raise NotImplementedError("Rendering of modality %s is not implemented!" % modality)
  return frame


# ---------- main ----------

if __name__ == '__main__':
  # parse input
  FLAGS = ARGPARSER.parse_args()
  print("Hello from the simulation recorder!")
  print("Arguments: ", FLAGS)

  # directory setup
  if not os.path.exists(FLAGS.record_path):
    os.makedirs(FLAGS.record_path)

  # load model
  model = mujoco_py.load_model_from_path(FLAGS.mjmodel_path)
  assets_file = 'assets.xml'
  assets_path = os.path.join(os.path.dirname(FLAGS.mjmodel_path), assets_file)
  assets_xml = ET.parse(assets_path).getroot()
  world_file = os.path.split(FLAGS.mjmodel_path)[1].replace('env', 'world')
  world_path = os.path.join(os.path.dirname(FLAGS.mjmodel_path), world_file)
  world_xml = ET.parse(world_path).getroot()

  # set recording intervals
  model_xml = ET.parse(FLAGS.mjmodel_path).getroot()
  timestep = float(model_xml.find("./option").attrib['timestep'])
  snapshot_interval = math.ceil((1.0 / FLAGS.fps) / timestep)
  total_frames = FLAGS.mjsim_time * FLAGS.fps
  total_steps = snapshot_interval * total_frames
  print("Simulation timestep: %.3fs" % timestep)
  print("Snapshot interval: %d" % snapshot_interval)
  print("Total simulation steps to perform: %d" % total_steps)
  print("Total frames to capture: %d" % total_frames)

  # create and initialize simulation
  sim = mujoco_py.MjSim(model)

  # load simulation state
  if FLAGS.mjsim_state_path:
    with open(FLAGS.mjsim_state_path, 'rb') as f:
      sim_state = pickle.load(f)
    sim.set_state(sim_state)

  if 'rgb' in FLAGS.formats:
    sim = _init_scene_rgb(
        sim, world_xml,
        FLAGS.lightid, FLAGS.walltex, FLAGS.floortex,
        FLAGS.color_mode)
  elif 'vseg' in FLAGS.formats:
    # parse violations from environment name
    env_str = os.path.basename(FLAGS.mjmodel_path)
    h_str = re.search(r'h=\d+', env_str).group(0)
    vcom_str = re.search(r'vcom=\d+', env_str).group(0)
    vpsf_str = re.search(r'vpsf=\d+', env_str).group(0)
    violations = []
    violations.append(int(vcom_str.lstrip('vcom=')))
    violations.append(int(vpsf_str.lstrip('vpsf=')))
    violations = set(violations)
    violations.remove(0) # no violation needs rendering
    height = int(h_str.lstrip('h='))
    sim = _init_scene_vseg(
        sim, world_xml,
        height, violations)
  elif 'iseg' in FLAGS.formats:
    env_str = os.path.basename(FLAGS.mjmodel_path)
    h_str = re.search(r'h=\d+', env_str).group(0)
    height = int(h_str.lstrip('h='))
    sim = _init_scene_iseg(
        sim, world_xml,
        height)

  # adjust cameras
  cm = CameraModder(sim)
  for cam_name in FLAGS.cameras:
    cx, cy, cz = cm.get_pos(cam_name)
    cm.set_pos(cam_name, (cx, cy, cz + FLAGS.cam_height_offset))

  # create rendering setups for modalities
  render_height = FLAGS.resolution[0]
  render_width = FLAGS.resolution[1]
  render_sims = {}
  for modality in FLAGS.formats:
    render_sims.update({modality : setup_render(sim, modality)})

  # run simulation and record screenshots
  stack_collapsed = False
  pcds = {}
  for camera in FLAGS.cameras:
      pcds[camera] = []
  for i in tqdm.tqdm(range(total_steps)):
    if i % snapshot_interval == 0 and i // snapshot_interval < FLAGS.max_frames:
      frame_nr = i // snapshot_interval
      for modality in FLAGS.formats:
        for camera in FLAGS.cameras:

          # render frame
          frame = render_modality(
              render_sims[modality], modality, camera,
              render_height, render_width, world_xml)

          # save frame
          if modality == 'depth':
            frame_mono = np.flip(frame[:, :render_width], 0)
          else:
            frame_mono = np.flip(frame[:, :render_width, :], 0)
          frame_fn = "%s-w=%s-f=%s-l=%s-c=%s-%s-mono-%s.%s" % \
              (modality, FLAGS.walltex, FLAGS.floortex, FLAGS.lightid, \
              FLAGS.color_mode, camera, frame_nr, FLAGS.file_format)
          scipy.misc.imsave(
              os.path.join(FLAGS.record_path, frame_fn),
              frame_mono)
          if FLAGS.with_stereo:
            frame_stereo = np.flip(frame, 0)
            frame_fn = "%s-w=%s-f=%s-l=%s-c=%s-%s-stereo-%s.%s" % \
                (modality, FLAGS.walltex, FLAGS.floortex, FLAGS.lightid, \
                FLAGS.color_mode, camera, frame_nr, FLAGS.file_format)
            scipy.misc.imsave(
                os.path.join(FLAGS.record_path, frame_fn),
                frame_stereo)
          if modality == 'depth':
            pcd = _convert_rgbd_to_pointcloud(sim, camera, render_height,
                                    render_width, world_xml, frame_mono)
            pcds[camera].append(pcd)

    if not stack_collapsed and i > BURN_IN_STEPS:
      velocities = np.abs(sim.data.sensordata)
      stack_collapsed = np.any(velocities > VELOCITY_TOLERANCE)
    sim.step()

    for modality in FLAGS.formats:
      render_sims[modality].step()

  with h5py.File(os.path.join(FLAGS.record_path, 'data.h5'), 'w') as F:
    F['stability'] = not stack_collapsed
    for camera in FLAGS.cameras:
        F[camera+'/pcd'] = np.array(pcds[camera])

  # print results
  print("Stack collapse: %s" % stack_collapsed)
