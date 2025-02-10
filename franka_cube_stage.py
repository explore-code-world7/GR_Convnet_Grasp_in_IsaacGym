# basic functions

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
from torch.utils import data
import random
import time
from PIL import Image
import os, imageio, shutil
import matplotlib.pyplot as plt 
from skimage.feature import peak_local_max


# import gc party

from inference.post_process import post_process_output
from utils.dataset_processing import grasp
from grtest.rgbd_dataset import Grasp_Rgbd_TestSet
from utils.dataset_processing.grasp import Grasp
from inference.models.grconvnet3 import GenerativeResnet

# 这是代表什么意思
def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def cube_grasping_yaw(q, corners):
    """ returns horizontal rotation required to grasp cube """
    rc = quat_rotate(q, corners)        # 空间向量，按单位四元数表示的矩阵，旋转得goal转向
    yaw = (torch.atan2(rc[:, 1], rc[:, 0]) - 0.25 * math.pi) % (0.5 * math.pi)      # 计算俯仰角
    theta = 0.5 * yaw       # tensor
    # print(f"theta={theta}")
    w = theta.cos()
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = theta.sin()
    yaw_quats = torch.stack([x, y, z, w], dim=-1)       # 相当于绕Z轴旋转yaw
        # since 绕z轴, K=(0,0,1), quaterion = (0, 0, cos (yaw/2), sin(yaw/2))
    return yaw_quats

# global的好处是，变量不用以形参的形式传递
def control_ik(dpose):
    global damping, j_eef, num_envs
    # solve damped least squares
    # print(f"j_eef_T.shape = {j_eef.shape}")   # j_eef.shape = (256, 6, 7), dpose = (256, 6)
    j_eef_T = torch.transpose(j_eef, 1, 2)          #(256, 7, 6)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 7)    # J^T(J@J^T+\lmbda)d_{pose}
    return u        # (256, link_dof=7)


def control_osc(dpose):
    global kp, kd, kp_null, kd_null, default_dof_pos_tensor, mm, j_eef, num_envs, dof_pos, dof_vel, hand_vel
    mm_inv = torch.inverse(mm)      # [256, 7, 7]
    m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)   
    m_eef = torch.inverse(m_eef_inv)
    u = torch.transpose(j_eef, 1, 2) @ m_eef @ (
        kp * dpose - kd * hand_vel.unsqueeze(-1))       # \tau_1 = J^T \ddot x_{des}=Kp(x_{target}-x)-K_D\dot x

    # Nullspace control torques `u_null` prevents large changes in joint configuration
    # They are added into the nullspace of OSC so that the end effector orientation remains constant
    # roboticsproceedings.org/rss07/p31.pdf
    j_eef_inv = m_eef @ j_eef @ mm_inv  # {J@M&{-1}@J}^{-1}@J@M^{-1}
    u_null = kd_null * -dof_vel + kp_null * (
        (default_dof_pos_tensor.view(1, -1, 1) - dof_pos + np.pi) % (2 * np.pi) - np.pi)
    u_null = u_null[:, :7]
    u_null = mm @ u_null
    u += (torch.eye(7, device=device).unsqueeze(0) - torch.transpose(j_eef, 1, 2) @ j_eef_inv) @ u_null     # tau2 = I-J^TJ
    return u.squeeze(-1)



# set random seed
np.random.seed(51)

torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface
gym = gymapi.acquire_gym()

# parse arguments

# Add custom arguments
custom_parameters = [
    {"name": "--controller", "type": str, "default": "ik",
     "help": "Controller to use for Franka. Options are {ik, osc}"},
    {"name": "--num_envs", "type": int, "default": 256, "help": "Number of environments to create"},
    {"name": "--test", "type":bool, "default": False, "help": "whether to implement test"},
    {"name": "--multiple", "type":bool, "default": False, "help": "whether to lay multiple objects"},
    {"name": "--real_po", "type":bool, "default": False, "help": "whether to use real position and orientation"},
]

args = gymutil.parse_arguments(
    description="Franka Jacobian Inverse Kinematics (IK) + Operational Space Control (OSC) Example",
    custom_parameters=custom_parameters,
)

print(args)
multi_object = args.multiple

# Grab controller
controller = args.controller
assert controller in {"ik", "osc"}, f"Invalid controller specified -- options are (ik, osc). Got: {controller}"

# set torch device
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

# Set controller parameters
# IK params
damping = 0.05

# OSC params
kp = 150.
kd = 2.0 * np.sqrt(kp)
kp_null = 10.
kd_null = 2.0 * np.sqrt(kp_null)

# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

asset_root = "./assets"

# create table asset
# table box的尺寸
table_dims = gymapi.Vec3(1.0, 1.0, 0.4)
# 调成[1.0, 1.0, 0.4]，即使increase spacing也没用，可能是超出franka的移动范围，franka被卡住了
# 是franka=[0,0,0], tables[0.5, 0, 0.5*height]
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

# create box asset
box_size = 0.045
asset_options = gymapi.AssetOptions()
box_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)

# load franka asset
franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.flip_visual_attachments = True
franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)

# configure franka dofs
franka_dof_props = gym.get_asset_dof_properties(franka_asset)
franka_lower_limits = franka_dof_props["lower"]
franka_upper_limits = franka_dof_props["upper"]
franka_ranges = franka_upper_limits - franka_lower_limits
franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)

# use position drive for all dofs
if controller == "ik":
    franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
    franka_dof_props["stiffness"][:7].fill(400.0)
    franka_dof_props["damping"][:7].fill(40.0)
else:       # osc
    franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
    franka_dof_props["stiffness"][:7].fill(0.0)
    franka_dof_props["damping"][:7].fill(0.0)
# grippers
franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
franka_dof_props["stiffness"][7:].fill(800.0)
franka_dof_props["damping"][7:].fill(40.0)

# default dof states and position targets
franka_num_dofs = gym.get_asset_dof_count(franka_asset)
default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
default_dof_pos[:7] = franka_mids[:7]   # 关节默认位置设置为中间值
# grippers open
default_dof_pos[7:] = franka_upper_limits[7:]   # gripper = [0.04, 0.04]，位置值=满开

default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos

# send to torch
default_dof_pos_tensor = to_torch(default_dof_pos, device=device)

# get link index of panda hand, which we will use as end effector
franka_link_dict = gym.get_asset_rigid_body_dict(franka_asset)
franka_hand_index = franka_link_dict["panda_hand"]
print(f"franka_link_dict={franka_link_dict}")
# print(f"franka_hand_index={franka_hand_index}")

# configure env grid
num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

franka_pose = gymapi.Transform()        # 有franka_pose.p, franka_pose.r = [0,0,0,1]
franka_pose.p = gymapi.Vec3( -0.1, 0, 0.2)

table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)    # 

box_pose = gymapi.Transform()

envs = []
table_idxs = []
box_idxs = []
hand_idxs = []
init_pos_list = []
init_rot_list = []

if multi_object:
    box2_pose = gymapi.Transform()
    box2_idxs = []

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# add camera binded to env
cams =  []
rgbd_tensors = []
depth_tensors = []
camera_width = 224
camera_height = 224


K_list = []
view_list = []
projct_list = []


for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add table
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)
    table_idx = gym.get_actor_rigid_body_index(env, table_handle, 0, gymapi.DOMAIN_SIM)
    table_idxs.append(table_idx)

    # add box
    # x,y,z轴分别是什么?
    box_pose.p.x = table_pose.p.x + np.random.uniform(-0.2, 0.1)
    box_pose.p.y = table_pose.p.y + np.random.uniform(-0.1, 0.2)
    # box_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.3)
    box_pose.p.z = table_dims.z + 0.5 * box_size
    box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
    # box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.pi/4)
    box_handle = gym.create_actor(env, box_asset, box_pose, "box", i, 0)
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    # get global index of box in rigid body state tensor
    box_idx = gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
    box_idxs.append(box_idx)

    if multi_object:
        print(multi_object)
        print("*"*100)
        # add box2
        for iter_time in range(1000):
            box2_pose.p.x = table_pose.p.x + np.random.uniform(-0.3, 0.2)
            box2_pose.p.y = table_pose.p.y + np.random.uniform(-0.2, 0.3)
            # box_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.3)
            box2_pose.p.z = table_dims.z + 0.5 * box_size

            to_dist = np.array([box_pose.p.x - box2_pose.p.x, box_pose.p.y - box2_pose.p.y, box_pose.p.z - box2_pose.p.z])
            if(np.linalg.norm(to_dist) > np.linalg.norm([box_size,box_size])):
                break
        box2_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
    
        box2_handle = gym.create_actor(env, box_asset, box2_pose, "box2", i, 0)        
        color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        gym.set_rigid_body_color(env, box2_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        box2_idx = gym.get_actor_rigid_body_index(env, box2_handle, 0, gymapi.DOMAIN_SIM)
        box2_idxs.append(box2_idx)

    # add franka
    franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", i, 2)

    # set dof properties
    gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

    # set initial dof states
    gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)

    # set initial position targets
    gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)

    # get inital hand pose
    hand_handle = gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
    hand_pose = gym.get_rigid_transform(env, hand_handle)
    init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
    # print(f"hand pose ={hand_pose.p.x, hand_pose.p.y, hand_pose.p.z}")
    init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

    # get global index of hand in rigid body state tensor
    hand_idx = gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
    hand_idxs.append(hand_idx)

    # add camera
    cam_props = gymapi.CameraProperties()
    cam_props.width = camera_width
    cam_props.height = camera_height
    cam_props.enable_tensors = True     # direct access of tensors without need to copy from GPU to CPU
    cam_handle = gym.create_camera_sensor(env, cam_props)
    cams.append(cam_handle)

    target_pos = gymapi.Vec3(table_pose.p.x, table_pose.p.y, table_dims.z)
    camera_pos = gymapi.Vec3(target_pos.x+0.01, target_pos.y, target_pos.z+0.4)
    gym.set_camera_location(cam_handle, env, camera_pos, target_pos)        # 设置在指定位置
        # 相机的正常朝向是(0.7, 0, 0, 0.7)，正向x轴

    # 计算相机的内参矩阵
    image_width = cam_props.width
    image_height = cam_props.height
    vertical_fov = (image_height / image_width * cam_props.horizontal_fov) * np.pi / 180
    horizontal_fov = cam_props.horizontal_fov*np.pi / 180

    f_x = (image_width / 2.0) / np.tan(horizontal_fov / 2.0)
    f_y = (image_height / 2.0) / np.tan(vertical_fov / 2.0)    

    # K = np.matrix(torch.tensor([[f_x, 0.0, image_width / 2.0], [0.0, f_y, image_height / 2.0], [0.0, 0.0, 1.0]], dtype=torch.float))
    # print(f"K ={K}")
    # # box_pos = (box_pose - cam_props)

    # # 输出相机的相关参数
    # tmp_cam_transform = gym.get_camera_transform(sim, env, cam_handle)
    # print(f"camera {i}, pos = {tmp_cam_transform.p}, pose = {tmp_cam_transform.r}")

    # tmp_cam_view = np.matrix(gym.get_camera_view_matrix(sim, env, cam_handle))
    # print(f"camera view = {tmp_cam_view}")

    tmp_cam_proj = np.matrix(gym.get_camera_proj_matrix(sim, env, cams[0]))
    print(f"camera proj = {tmp_cam_proj}")

    # cache rgbd and depth images
    rgbd_tensor = gym.get_camera_image_gpu_tensor(sim, env, cam_handle, gymapi.IMAGE_COLOR)
    torch_rgbd_tensor = gymtorch.wrap_tensor(rgbd_tensor)
    rgbd_tensors.append(torch_rgbd_tensor)

    depth_tensor = gym.get_camera_image_gpu_tensor(sim, env, cam_handle, gymapi.IMAGE_DEPTH)
    torch_depth_tensor = gymtorch.wrap_tensor(depth_tensor)
    depth_tensors.append(torch_depth_tensor)

# point camera at middle env
cam_pos = gymapi.Vec3(4, 3, 2)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
    # camera也分 viewer_camera & (env) camera
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)
# gym.viewer_camera_look_at(viewer, middle_env, gymapi.Vec3(-10, 0, 0.5), gymapi.Vec3(2.0, 0, 0.5))


# ==== prepare tensors =====
# from now on, we will use the tensor API that can run on CPU or GPU
gym.prepare_sim(sim)

# initial hand position and orientation tensors
init_pos = torch.Tensor(init_pos_list).view(num_envs, 3).to(device)
init_rot = torch.Tensor(init_rot_list).view(num_envs, 4).to(device)

# hand orientation for grasping
down_q = torch.stack(num_envs * [torch.tensor([1.0, 0.0, 0.0, 0.0])]).to(device).view((num_envs, 4))    # v=0,无旋转
    # 绕x轴旋转180’

# box corner coords, used to determine grasping yaw+
box_half_size = 0.5 * box_size
corner_coord = torch.Tensor([box_half_size, box_half_size, box_half_size])
corners = torch.stack(num_envs * [corner_coord]).to(device)

# downard axis
down_dir = torch.Tensor([0, 0, -1]).to(device).view(1, 3)

# get jacobian tensor
# for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
# 10 = link (velocities) nums, 9 = dof nums, 6= linear + angular velocity
# it seems as link velocity = (link num, dof num)*(dof velocity), Jacobian transform
# if freed, base has 6 dofs, which is mobile, Jacobian matrix has (num_envs, 11, 6, 15)
_jacobian = gym.acquire_jacobian_tensor(sim, "franka")
jacobian = gymtorch.wrap_tensor(_jacobian)
print(f"jacobian's shape = {jacobian.shape}")

# jacobian entries corresponding to franka hand
j_eef = jacobian[:, franka_hand_index - 1, :, :7]   # just consider transform of
    # joints = hand from motion
    # links = franka_hand

# get mass matrix tensor, [num_envs, dof_num, dof_num]
# Euler equation: N=CI\dot w +w\times CIw;      Force~Angular velocity
_massmatrix = gym.acquire_mass_matrix_tensor(sim, "franka")
mm = gymtorch.wrap_tensor(_massmatrix)
mm = mm[:, :7, :7]          # only need elements corresponding to the franka arm

# get rigid body state tensor
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# print actor list
actor_dict = {}
for i in range(gym.get_actor_count(envs[0])):
    actor_dict[i] = gym.get_actor_name(envs[0], i)
print(f"actor dict = {actor_dict}")

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
# 查询dof names
table_dof = gym.get_actor_dof_dict(envs[0], 0)
box_dof = gym.get_actor_dof_dict(envs[0], 1)
franka_dof = gym.get_actor_dof_dict(envs[0], 2)
# print("table")
# print(table_dof)
# print("box")
# print(box_dof)
# but how many dofs are there?
print(f"franka's dof_dict = {franka_dof}")
# each dof has position and velocity variables
print(f"franka's dof property = {gym.get_actor_dof_properties(envs[0], 2)}")        # 输出每个自由度的值
# ( True, -2.8973,  2.8973, 1, 2.175, 87., 400., 40., 0., 0.)
# hasLimits, lower, upper, driveMode, stiffness, damping, velocity, effort, friction, armature

# 对所有env的dof一并操作
# each dof has 2 components, position and velocity
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_pos = dof_states[:, 0].view(num_envs, 9, 1)     # [256, 9, 1]
dof_vel = dof_states[:, 1].view(num_envs, 9, 1)
# Create a tensor noting whether the hand should return to the initial position
hand_restart = torch.full([num_envs], False, dtype=torch.bool).to(device)

# Set action tensors
pos_action = torch.zeros_like(dof_pos).squeeze(-1)
pos_action[:,7:9] = 0.04        # 初始张开
effort_action = torch.zeros_like(pos_action)



# control class
# multi-stage control in nut-bolt
class GraspFSM:

    def __init__(self, num_envs):
        self._num_envs = num_envs
        self.grasp_offset = 0.11 if controller == "ik" else 0.10
        # self._state = ["query_box"]*num_envs
        self._state = torch.zeros(num_envs).to(device)
        # 0: query_box
        # 1: return_origin

    # returns
    # 我认为的封装，就是把初始变量放在构造类中处理
    def update_zero(self, dof_pos, box_pos, hand_pos):
            #    nut_pose, bolt_pose, hand_pose, current_gripper_sep):
        newState = self._state

        # 进行0状态的更新
        to_box = box_pos - hand_pos
        box_dist = torch.norm(to_box, dim=-1).unsqueeze(-1)
        gripper_sep = dof_pos[:, 7] + dof_pos[:, 8]
        gripped = (gripper_sep < 0.045) & (box_dist < self.grasp_offset + 0.5 * box_size)    # 抓手和box足够近            
        # print(f"gripped = {gripped}")
        # newState = torch.where( (self._state == "query_box") & gripped, ["return_origin"]*num_envs, ["query_box"]*num_envs)
        # newState[(self._state == "query_box") & gripped ] = "return_origin"
        # newState = torch.where( (self._state == 0) & gripped, torch.ones(self._num_envs,1).to(device), torch.zeros(self._num_envs,1).to(device))
        begin_return = ((self._state == 0) & gripped.squeeze(-1))   # gripped:[256,1],  begin_return: [256]
        newState[begin_return] = 1     
        # print(f"status1 = {newState}")
        self._state = newState                

    def update_one(self, hand_pos, init_pos):
        newState = self._state
        # 进行1状态的更新
        to_init = init_pos - hand_pos       # [256,3]
        # print(f"to_init.shape = {to_init.shape}")
        init_dist = torch.norm(to_init, dim=-1).squeeze(-1)     # take norm on dim=-1, get [256,1]
        # return_to_start = (init_dist < 0.02)
        # print(init_dist.shape)
        begin_query = ((init_dist < 0.02) & (self._state == 1)).squeeze(-1)
        # newState = torch.where( (self._state == "return_origin") & return_to_start, ["query_box"]*num_envs, ["return_origin"]*num_envs)
        # newState[(self._state == "return_origin") & return_to_start] = "query_box"
        # newState = torch.where( (self._state == 1) & return_to_start, torch.zeros(self._num_envs,1).to(device), torch.ones(self._num_envs,1).to(device))
        newState[begin_query] = 0
        # print(f"status2 = {newState}")
        self._state = newState                
        

grasp_offset = 0.11 if controller == "ik" else 0.10
grasp_agent = GraspFSM(num_envs)


# create camera folder
img_dir = "franka_img"
if os.path.exists(img_dir):
    shutil.rmtree(img_dir)
os.mkdir(img_dir)


# draw picture
fig, ax = plt.subplots()
predict_dir = "box_predict"
if os.path.exists(predict_dir):
    shutil.rmtree(predict_dir)
os.mkdir(predict_dir)

# import model
device = torch.device("cuda:0")
gcnet = torch.load("logs/250131_2338_training_cornell/epoch_44_iou_0.96").to(device)
gcnet.eval()


# record detect values
box_pos = torch.zeros((num_envs, 3)).to(device)
box_rot = torch.zeros((num_envs, 4)).to(device)

table_pos = torch.zeros((num_envs, 3)).to(device)

# angles和centres仅仅存储图像识别的中间结果,还要转换成真实坐标和旋转四元数, 在执行过程用中间变量即可,box_pos和box_rot取代他们
# centres = torch.zeros((num_envs, 2)).to(device)
# angles = torch.zeros((num_envs, 1)).to(device)

restart = torch.tensor([True]*num_envs)


def take_photo():
    global gym, rgbd_tensors, depth_tensors, gcnet, fig, ax, frame_no, centres, angles, box_pos, box_rot, real_box_pos, real_box_rot,multi_object 
    global table_dims, box_size
    global total_grasp_time, x_diff, y_diff, angle_diff, perform_test
    if multi_object:
        global real_box2_pos, real_box2_rot

    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)
        
    for i in range(num_envs):
        # write tensor to image
        fname = os.path.join(img_dir, "cam-%04d-%04d.png" % (frame_no, i))
        cam_img = rgbd_tensors[i].cpu().numpy()
        cam_img[cam_img == 165] = 255
        imageio.imwrite(fname, cam_img)

        # depfname = os.path.join(img_dir, "dep-%04d-%04d.jpg" % (frame_no, i))
        # depth_img = depth_tensors[i].cpu().numpy()
        # print(f"depth max = {np.max(depth_img)}, min = {np.min(depth_img)}")
        # imageio.imwrite(depfname, depth_img)

        # 不括号，会判断frame_no == (0|restart[i])
        if (frame_no == 1) | restart[i]:
            print()
            print(f"for env {i}")
            if perform_test:
                total_grasp_time -=1
            # print(f"total_grasp_time ={total_grasp_time}")
            com_img = torch.zeros(224, 224, 4).to(device)
            # cam_img = cam_tensors[i]
            # depth_img = depth_tensors[i]
            rgb_img = rgbd_tensors[i]        #rgbd
            depth_img = depth_tensors[i]
            depth_img2 = depth_img.cpu().numpy()       #这创建了一个新的对象,不必先clone()
            # -inf implies no depth value, set it to zero. output will be black.
            depth_img2[depth_img2 == -np.inf] = 0

            # clamp depth image to 10 meters to make output image human friendly
            depth_img2[depth_img2 < -10] = -10
            norm_depth_img = 255.0*(depth_img2/np.min(depth_img2 - 1e-4))
            normalized_depth_image = Image.fromarray(norm_depth_img.astype(np.uint8), mode="L")
            normalized_depth_image.save(os.path.join(predict_dir,f'frameno_{frame_no}_env_{i}.jpg'))
            np.savetxt(os.path.join(predict_dir,f'frameno_{frame_no}_env_{i}.txt'), depth_img.cpu().numpy(), fmt='%f')  # fmt='%d' 表示以整数格式写入
            # print(f"depth max = {depth_img.max().item()}, min = {depth_img.min().item()}")

            com_img[:,:,1:] = rgb_img[:,:,:3]
            # # remove shadow
            com_img[com_img == 165] = 255
            com_img[:,:,0] = depth_img
            com_img = com_img.permute(2,0,1).unsqueeze(0)       # [1, 4, 224, 224]
            # print(f"com_img.shape = {com_img.shape}, depth_img = {depth_img.shape}")

            com_img = com_img.to(torch.float32).to(device)
            with torch.no_grad():
                q_img, cos_img, sin_img, width_img = gcnet(com_img)

            q_img, ang_img, width_img = post_process_output(q_img, cos_img, sin_img, width_img)     # [1, 1, 224, 224]

            # g_img的shape = [224,224]

            if(len(q_img.shape)==2):        # resist to squeeze's side effect
                q_img = np.expand_dims(q_img, 0)
                ang_img = np.expand_dims(ang_img, 0)
                width_img = np.expand_dims(width_img, 0)
            
            # print(f"len of q_img = {q_img.shape}, {ang_img.shape}, {width_img.shape}")

            for img_idx in range(q_img.shape[0]):       # batch_size = q_img.shape[0]
                grasps = grasp.detect_grasps(q_img[img_idx], ang_img[img_idx], np.ones((224, 224))*6, no_grasps=1)
                # print(f"grasps = {grasps[0].shape}")

                # print(f"len of grasps ={len(grasps)}")
                centres = torch.tensor(grasps[0].center).to(device)
                angles = torch.tensor(grasps[0].angle).to(device)      # 这里只会改变angles with id =0 

                # print(f"centre coordinate = {centres[img_idx]}")

                # 索引不能是tensor，转为item()再转为int
                img_y = int(centres[0].item()); img_x = int(centres[1].item());   rgb_val = rgb_img[img_y][img_x]
                # print(f"target depth ={depth_img[img_y][img_x]}, rgb = {rgb_val[0]}, {rgb_val[1]}, {rgb_val[2]}")

                # 利用内外参变换，从图像坐标转为空间坐标
                    # envs是第i个环境,不是随便一个环境
                vinv = np.linalg.inv(np.matrix(gym.get_camera_view_matrix(sim, envs[i], cams[i])))
                proj = gym.get_camera_proj_matrix(sim, envs[i], cams[i])

                # env_position = gym.get_env_origin(envs[i])
                # env_to_global = np.identity(4)
                # env_to_global[:3,3] = np.array([env_position.x, env_position.y, env_position.z])

                fu = 2/proj[0, 0]
                fv = 2/proj[1, 1]

                u = -(img_x - image_width/2)/image_width
                v = (img_y - image_height/2)/image_height
                d = depth_img[img_y, img_x].cpu().item()
                X2 = [d*fu*u, d*fv*v, d, 1]
                # print(f"X2 = {X2}")
                p2 = X2*vinv
                
                env_position = gym.get_env_origin(envs[i])
                # print(f"env origin = {env_position}, p2 = {p2}")
                env_to_global = np.identity(4)
                env_to_global[3,:3] = np.array([env_position.x, env_position.y, env_position.z])
                p2_env = p2 @np.linalg.inv(env_to_global)            
    
                real_box_pos_np = real_box_pos.cpu().numpy()[i]
                real_box_rot_np = real_box_rot.cpu().numpy()[i]

                print(f"Predicted box's world coordinate = {p2_env},\nReal box position = {real_box_pos_np}") 

                # take photo输出pos&angle转化
                box_pos[i,:2] = torch.tensor(p2_env[0,:2])
                box_pos[i, 2] = table_dims.z + 0.5*box_size

                rot_quat = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), angles.cpu().numpy())
                # print(f"size of real box rotation is {real_box_pos_np.shape}")
                print(f"angle = {angles.cpu().numpy()}, predicted rotation = {rot_quat},\nreal box rotation = {real_box_rot_np}")

                if multi_object:               
                    real_box2_pos_np = real_box2_pos.cpu().numpy()[i]
                    real_box2_rot_np = real_box2_rot.cpu().numpy()[i]

                    if(np.linalg.norm(p2_env[0,:2]- real_box_pos_np[:2])<np.linalg.norm(p2_env[0,:2] - real_box2_pos_np[:2])):
                        x_diff += np.abs(p2_env[0,0] - real_box_pos_np[0])                    
                        y_diff += np.abs(p2_env[0,1] - real_box_pos_np[1])
                        angle_diff += np.abs(angles.cpu().numpy()%(np.pi/2)- np.arctan2(real_box_rot_np[2], real_box_rot_np[3])%(np.pi/2))
                    else:
                        x_diff += np.abs(p2_env[0,0] - real_box2_pos_np[0])                    
                        y_diff += np.abs(p2_env[0,1] - real_box2_pos_np[1])
                        angle_diff += np.abs(angles.cpu().numpy()%(np.pi/2)-np.arctan2(real_box2_rot_np[2], real_box2_rot_np[3])%(np.pi/2))

                    print(f"Real box2 position = {real_box2_pos_np}")
                    print(f"Real box2 rotation = {real_box2_rot_np}")

                else:
                    x_diff += np.abs(p2_env[0,0] - real_box_pos_np[0])                    
                    y_diff += np.abs(p2_env[0,1] - real_box_pos_np[1])
                    angle_diff += np.abs(angles.cpu().numpy()%(np.pi/2)- np.arctan2(real_box_rot_np[2], real_box_rot_np[3])%(np.pi/2))

                box_rot[img_idx, :] = torch.tensor([rot_quat.x, rot_quat.y, rot_quat.z, rot_quat.w], dtype=torch.float32).to(device)
                # box_rot[img_idx, ]
                for gid, _grasp in enumerate(grasps):

                    print(_grasp.center, _grasp.angle)
                    plane = np.zeros((224, 224))
                    plane[_grasp.center[0],_grasp.center[1]]=1
                    plt.imshow(plane)
                    fig.savefig(f"box_predict/image_{frame_no}_{gid}.png")

    gym.end_access_image_tensors(sim)

perform_test = args.test
total_grasp_time = 100
total_grasp_time2 = 200
x_diff = 0.0
y_diff = 0.0
angle_diff = 0.0

# simulation loop
while not gym.query_viewer_has_closed(viewer):

    frame_no = gym.get_frame_count(sim)
    # print(f"frame_no = {frame_no}")

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)

    if frame_no == 0:
        # 发现frame_no==0，只有env0的图像是正常的，因此把frame_no=0打patch
        pass
    else:
        # acquire box and hand positions
        # 不同env的box拼接

        real_box_pos = rb_states[box_idxs, :3]       # [num_envs, 3]
        real_box_rot = rb_states[box_idxs, 3:7]      # [num_envs, 4]
        if multi_object:
            real_box2_pos = rb_states[box2_idxs, :3]       # [num_envs, 3]
            real_box2_rot = rb_states[box2_idxs, 3:7]      # [num_envs, 4]
        # print(f"real box_pos = {real_box_pos}")
        # print(f"box_pos.r = {box_pose.r}")
        # print(f"box_rot = {box_rot[0]}")

        # print(f"rb_states.shape = {rb_states.shape}")
        # print(f"box_pos.shape ={box_pos.shape}, box_rot.shape = {box_rot.shape}")

        table_pos = rb_states[table_idxs, :3]
        hand_pos = rb_states[hand_idxs, :3]
        hand_rot = rb_states[hand_idxs, 3:7]
        hand_vel = rb_states[hand_idxs, 7:]     # rigid body states表示asset各个link的位姿速度；
                                                # root body states表示asset原点的位姿速度
        if frame_no == 1:
            take_photo()
            grasp_agent.update_zero(dof_pos, box_pos, hand_pos)
            # print(f"real box pos = {real_box_pos}, table_pos = {table_pos}")
        else:
            current_state = grasp_agent._state.clone()
            grasp_agent.update_one(hand_pos, init_pos)
            new_state = grasp_agent._state.clone()
            restart = (current_state == 1) & (new_state == 0)
            take_photo()
            grasp_agent.update_zero(dof_pos, box_pos, hand_pos)   # 之前更新过的不能再更新了，但在本题不会出现这种情形
            # 对current_state==0多个&即可
        
        if perform_test:
            # print(f"total_grasp_time = {total_grasp_time}")
            if(total_grasp_time <=0):
                break

        # hand control
        to_box = box_pos - hand_pos
        box_dist = torch.norm(to_box, dim=-1).unsqueeze(-1)
        box_dir = to_box / box_dist
        box_dot = box_dir @ down_dir.view(3, 1)     # 手与box的方向是否逼近竖直向下

        # current_state = grasp_agent._state.clone()
        # # box每次restart拍摄一次，一直沿用到下一周期restart
        # grasp_agent.update(dof_pos, box_pos, hand_pos, init_pos)
        # new_state = grasp_agent._state.clone()
        # restart = (current_state == 1) & (new_state == 0)

        yaw_q = cube_grasping_yaw(box_rot, corners)     # corners=[z/2, z/2, z/2]的quaterion
        box_yaw_dir = quat_axis(yaw_q, 0)           # 将[1,0,0]旋转yaw_q的方向,得到box坐标系下的yaw轴方向向量
        hand_yaw_dir = quat_axis(hand_rot, 0)       # 将[1,0,0]旋转手的朝向，得到hand坐标系下的yaw轴方向向量
        yaw_dot = torch.bmm(box_yaw_dir.view(num_envs, 1, 3), hand_yaw_dir.view(num_envs, 3, 1)).squeeze(-1)

        above_box = ((box_dot >= 0.99) & (yaw_dot >= 0.95) & (box_dist < grasp_offset * 3)).squeeze(-1)
            #   手在box正上方       手的朝向和抓取需要的朝向一致    手和box的距离足够近
        grasp_pos = box_pos.clone()
        grasp_pos[:, 2] = torch.where(above_box, box_pos[:, 2] + grasp_offset, box_pos[:, 2] + grasp_offset * 2.5)    

        grasp_rot = quat_mul(down_q, quat_conjugate(yaw_q))

        tmp_pos = torch.stack([grasp_agent._state]*3, dim=1)        # = (dim=-1), dim=0表示[3,256]
        goal_pos = torch.where(tmp_pos == 0, grasp_pos, init_pos)
        tmp_rot = torch.stack([grasp_agent._state]*4, dim=1)        # = (dim=-1), dim=0表示[3,256]
        goal_rot = torch.where(tmp_rot == 0, grasp_rot, init_rot)

        pos_err = goal_pos - hand_pos
        orn_err = orientation_error(goal_rot, hand_rot)
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

        if controller == "ik":
            pos_action[:, :7] = dof_pos.squeeze(-1)[:, :7] + control_ik(dpose)
        # else:       # osc
            effort_action[:, :7] = control_osc(dpose)

        # finger control
        # 1: when to grasp    
        tighten = ((box_dist < grasp_offset + 0.02).squeeze(-1)) & ((grasp_agent._state == 0))  # [256, 1]

        # torch赋值都是按最后一维赋值的
        pos_action[ tighten , 7:9] = torch.tensor([0.0, 0.0]).to(device)

        # 2: when to loose
        # if status = return and height >0.6
        loosen = ((hand_pos[:, 2] > 0.6).squeeze(-1)) & ((grasp_agent._state == 1))
        pos_action[ loosen, 7:9] = torch.tensor([0.04, 0.04]).to(device)
        
        # Deploy actions
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(effort_action))

    # update viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)



if perform_test:
    test_time = total_grasp_time2 - total_grasp_time
    print(f"test time = {test_time}")
    print(f"average x_diff = {x_diff/test_time}")
    print(f"average y_diff = {y_diff/test_time}")
    print(f"average angle_diff = {angle_diff/test_time}")