#!/usr/bin/env python

from __future__ import division

import pprint
import argparse
import eval_util, sim_util, util
import tps_registration as tps_registration_old
from rapprentice import tps_registration, planning
 
from rapprentice import registration, colorize, berkeley_pr2, \
     animate_traj, ros2rave, plotting_openrave, task_execution, \
     tps, func_utils, resampling, ropesim, rope_initialization, clouds
from rapprentice import math_utils as mu
from rapprentice.yes_or_no import yes_or_no
import pdb, time
 
import trajoptpy, openravepy
import rope_qlearn
from rope_qlearn import get_closing_pts, get_closing_inds
from knot_classifier import isKnot as is_knot, calculateCrossings
import os, os.path, numpy as np, h5py
from numpy import asarray
import atexit
import importlib
from itertools import combinations
import IPython as ipy
import random
import hashlib

COLLISION_DIST_THRESHOLD = 0.0
MAX_ACTIONS_TO_TRY = 10  # Number of actions to try (ranked by cost), if TrajOpt trajectory is infeasible
TRAJOPT_MAX_ACTIONS = 5  # Number of actions to compute full feature (TPS + TrajOpt) on
WEIGHTS = np.array([-1]) 
DS_SIZE = .025

class GlobalVars:
    unique_id = 0
    actions = None
    actions_cache = None
    tps_errors_top10 = []
    trajopt_errors_top10 = []
    actions_ds_clouds = {}
    rope_nodes_crossing_info = {}

def get_action_cloud(sim_env, action, args_eval):
    rope_nodes = get_action_rope_nodes(sim_env, action, args_eval)
    cloud = ropesim.observe_cloud(rope_nodes, sim_env.sim.rope_params.radius, upsample_rad=args_eval.upsample_rad)
    return cloud

def get_action_cloud_ds(sim_env, action, args_eval):
    if args_eval.downsample:
        if action not in GlobalVars.actions_ds_clouds:
            GlobalVars.actions_ds_clouds[action] = clouds.downsample(get_action_cloud(sim_env, action), DS_SIZE)
        return GlobalVars.actions_ds_clouds[action]
    else:
        return get_action_cloud(sim_env, action, args_eval)

def get_action_rope_nodes(sim_env, action, args_eval):
    rope_nodes = GlobalVars.actions[action]['cloud_xyz'][()]
    return ropesim.observe_cloud(rope_nodes, sim_env.sim.rope_params.radius, upsample=args_eval.upsample)

def color_cloud(xyz, endpoint_inds, endpoint_color = np.array([0,0,1]), non_endpoint_color = np.array([1,0,0])):
    xyzrgb = np.zeros((len(xyz),6))
    xyzrgb[:,:3] = xyz  
    xyzrgb[endpoint_inds,3:] = np.tile(endpoint_color, (endpoint_inds.sum(), 1))
    xyzrgb[~endpoint_inds,3:] = np.tile(non_endpoint_color, ((~endpoint_inds).sum(), 1))
    return xyzrgb

def redprint(msg):
    print colorize.colorize(msg, "red", bold=True)

def yellowprint(msg):
    print colorize.colorize(msg, "yellow", bold=True)

def draw_grid(sim_env, old_xyz, f, color = (1,1,0,1)):
    grid_means = .5 * (old_xyz.max(axis=0) + old_xyz.min(axis=0))
    grid_mins = grid_means - (old_xyz.max(axis=0) - old_xyz.min(axis=0))
    grid_maxs = grid_means + (old_xyz.max(axis=0) - old_xyz.min(axis=0))
    return plotting_openrave.draw_grid(sim_env.env, f.transform_points, grid_mins, grid_maxs, xres = .1, yres = .1, zres = .04, color = color)

def draw_axis(sim_env, hmat):
    handles = []
    handles.append(sim_env.env.drawarrow(hmat[:3,3], hmat[:3,3]+hmat[:3,0]/10.0, 0.005, (1,0,0,1)))
    handles.append(sim_env.env.drawarrow(hmat[:3,3], hmat[:3,3]+hmat[:3,1]/10.0, 0.005, (0,1,0,1)))
    handles.append(sim_env.env.drawarrow(hmat[:3,3], hmat[:3,3]+hmat[:3,2]/10.0, 0.005, (0,0,1,1)))
    return handles

def draw_finger_pts_traj(sim_env, flr2finger_pts_traj, color):
    handles = []
    for finger_lr, pts_traj in flr2finger_pts_traj.items():
        for pts in pts_traj:
            handles.append(sim_env.env.drawlinestrip(np.r_[pts, pts[0][None,:]], 1, color))
    return handles

def register_tps(sim_env, state, action, args_eval, interest_pts = None, closing_hmats = None, closing_finger_pts = None, reg_type='segment'):
    old_cloud = get_action_cloud(sim_env, action, args_eval)
    new_cloud = state[1]
    if reg_type == 'segment':
        old_rope_nodes = get_action_rope_nodes(sim_env, action, args_eval)
        state_id, new_cloud, new_rope_nodes = state
        def plot_cb(rope_nodes0, rope_nodes1, cloud0, cloud1, corr_nm, corr_nm_aug, f, pts_segmentation_inds0, pts_segmentation_inds1):
            from rapprentice.plotting_plt import plot_tps_registration_segment_proj_2d
            import matplotlib.pyplot as plt
            plot_tps_registration_segment_proj_2d(rope_nodes0, rope_nodes1, cloud0, cloud1, corr_nm, corr_nm_aug, f, pts_segmentation_inds0, pts_segmentation_inds1)
            
            if closing_hmats is not None:
                plt.subplot(223, aspect='equal')
                interest_pts_inds = np.zeros(len(cloud0), dtype=bool)
                for hmat in closing_hmats.values():
                    interest_pts_inds += np.apply_along_axis(np.linalg.norm, 1, cloud0 - hmat[:3,3]) < 0.025
                cloud1_resampled = corr_nm_aug.dot(cloud1)
                plt.scatter(cloud1_resampled[interest_pts_inds,0], cloud1_resampled[interest_pts_inds,1], c='none', edgecolors='b', marker='o', s=15)
                warped_cloud0 = f.transform_points(cloud0)
                plt.scatter(warped_cloud0[interest_pts_inds,0], warped_cloud0[interest_pts_inds,1], c='none', edgecolors='g', marker='o', s=15)
            
            if interest_pts is not None and len(interest_pts) > 0:
                plt.subplot(221, aspect='equal')
                pts = np.array(interest_pts)
                plt.scatter(pts[:,0], pts[:,1], marker='x')
                plt.subplot(222, aspect='equal')
                pts = f.transform_points(pts)
                plt.scatter(pts[:,0], pts[:,1], marker='x')
            if closing_hmats is not None:
                plt.subplot(221, aspect='equal')
                for hmat in closing_hmats.values():
                    plt.arrow(hmat[0,3], hmat[1,3], hmat[0,0]/10.0, hmat[1,0]/10.0, fc='r', ec='r')
                    plt.arrow(hmat[0,3], hmat[1,3], hmat[0,1]/10.0, hmat[1,1]/10.0, fc='g', ec='g')
                    plt.arrow(hmat[0,3], hmat[1,3], hmat[0,2]/10.0, hmat[1,2]/10.0, fc='b', ec='b')
                for i_plot in range(2,5):
                    plt.subplot(2,2,i_plot, aspect='equal')
                    for hmat in f.transform_hmats(np.array(closing_hmats.values())):
                        plt.arrow(hmat[0,3], hmat[1,3], hmat[0,0]/10.0, hmat[1,0]/10.0, fc='r', ec='r')
                        plt.arrow(hmat[0,3], hmat[1,3], hmat[0,1]/10.0, hmat[1,1]/10.0, fc='g', ec='g')
                        plt.arrow(hmat[0,3], hmat[1,3], hmat[0,2]/10.0, hmat[1,2]/10.0, fc='b', ec='b')
            if closing_finger_pts is not None:
                import matplotlib
                plt.subplot(221, aspect='equal')
                lines = []
                for pts_list in closing_finger_pts.values():
                    for pts in pts_list:
                        lines.append(np.r_[pts, pts[0][None,:]][:,:2])
#                         lines.append(pts[np.array([True,False,False,True])][:,:2])
                lc = matplotlib.collections.LineCollection(lines, colors=(1,0,0), lw=2)
                ax = plt.gca()
                ax.add_collection(lc)

                lines = []
                for pts_list in closing_finger_pts.values():
                    for i_pt in range(4):
                        closing_pts = mu.interp2d(np.linspace(0,1,20), np.arange(2), np.r_[pts_list[0][i_pt,:][None,:], pts_list[1][3-i_pt,:][None,:]])
                        lines.append(closing_pts[:,:2])
                lc = matplotlib.collections.LineCollection(lines, colors=(1,0,0), lw=1)
                ax = plt.gca()
                ax.add_collection(lc)

#                 closing_pts = []
#                 for pts_list in closing_finger_pts.values():
#                     for i_pt in range(4):
#                         closing_pts.append(mu.interp2d(np.linspace(0,1,10), np.arange(2), np.r_[pts_list[0][i_pt,:][None,:], pts_list[1][3-i_pt,:][None,:]]))
#                 closing_pts = np.concatenate(np.asarray(closing_pts))
#                 plt.scatter(closing_pts[:,0], closing_pts[:,1], c=(1,0,0), edgecolors=(1,0,0), marker=',', s=1)

                lines = []
                for pts_list in closing_finger_pts.values():
                    for pts in pts_list:
                        pts = f.transform_points(pts)
                        lines.append(np.r_[pts, pts[0][None,:]][:,:2])
                for i_plot in range(2,5):
                    plt.subplot(2,2,i_plot, aspect='equal')
                    lc = matplotlib.collections.LineCollection(lines, colors=(0,1,0), lw=2)
                    ax = plt.gca()
                    ax.add_collection(lc)

                lines = []
                for pts_list in closing_finger_pts.values():
                    for i_pt in range(4):
                        closing_pts = mu.interp2d(np.linspace(0,1,20), np.arange(2), np.r_[pts_list[0][i_pt,:][None,:], pts_list[1][3-i_pt,:][None,:]])
                        closing_pts = f.transform_points(closing_pts)
                        lines.append(closing_pts[:,:2])
                for i_plot in range(2,5):
                    plt.subplot(2,2,i_plot, aspect='equal')
                    lc = matplotlib.collections.LineCollection(lines, colors=(0,1,0), lw=1)
                    ax = plt.gca()
                    ax.add_collection(lc)
            plt.show()
        x_weights = np.ones(len(old_cloud)) * 1.0/len(old_cloud)
        f, corr = tps_registration.tps_segment_registration(old_rope_nodes, new_rope_nodes, 
                                                        cloud0 = old_cloud, cloud1 = new_cloud, corr_tile_pattern = np.eye(args_eval.upsample_rad), 
                                                        reg=np.array([0.00015, 0.00015, 0.0015]), x_weights=x_weights, plotting=False, plot_cb=plot_cb)
        if corr is not None:
            corr = tps_registration.tile(corr, np.eye(args_eval.upsample_rad))
        
    elif reg_type == 'rpm':
        vis_cost_xy = tps_registration.ab_cost(old_cloud, new_cloud) if args_eval.use_color else None
        f, corr = tps_registration.tps_rpm(old_cloud[:,:3], new_cloud[:,:3], vis_cost_xy=vis_cost_xy, user_data={'old_cloud':old_cloud, 'new_cloud':new_cloud})
    elif reg_type == 'bij':
        vis_cost_xy = tps_registration.ab_cost(old_cloud, new_cloud) if args_eval.use_color else None
        x_nd = old_cloud[:,:3]
        y_md = new_cloud[:,:3]
        scaled_x_nd, src_params = registration.unit_boxify(x_nd)
        scaled_y_md, targ_params = registration.unit_boxify(y_md)
        f,g = registration.tps_rpm_bij(scaled_x_nd, scaled_y_md, rot_reg=np.r_[1e-4, 1e-4, 1e-1], n_iter=50, reg_init=10, reg_final=.1, outlierfrac=1e-2, vis_cost_xy=vis_cost_xy)
        f = registration.unscale_tps(f, src_params, targ_params)
        corr = None
    return f, corr

def register_tps_cheap(sim_env, state, action, args_eval, reg_type='segment'):
    old_cloud = get_action_cloud(sim_env, action, args_eval)
    new_cloud = state[1]
    if reg_type == 'segment':
        old_rope_nodes = get_action_rope_nodes(sim_env, action, args_eval)
        old_rope_nodes_hash = hashlib.sha1(old_rope_nodes).hexdigest()
        if action not in GlobalVars.rope_nodes_crossing_info:
            if action not in GlobalVars.actions_cache:
                action_group = GlobalVars.actions_cache.create_group(action)
            else:
                action_group = GlobalVars.actions_cache[action]
            if old_rope_nodes_hash not in action_group:
                action_rope_nodes_group = action_group.create_group(old_rope_nodes_hash)
                crossings, crossings_links_inds, cross_pairs, rope_closed = calculateCrossings(old_rope_nodes)
                action_rope_nodes_group['rope_nodes'] = old_rope_nodes
                if crossings: action_rope_nodes_group['crossings'] = crossings
                if crossings_links_inds: action_rope_nodes_group['crossings_links_inds'] = crossings_links_inds
                if cross_pairs: action_rope_nodes_group['cross_pairs'] = list(cross_pairs)
                action_rope_nodes_group['rope_closed'] = rope_closed
            else:
                action_rope_nodes_group = action_group[old_rope_nodes_hash]
                assert np.all(old_rope_nodes == action_rope_nodes_group['rope_nodes'][()])
                crossings =  action_rope_nodes_group['crossings'][()] if 'crossings' in action_rope_nodes_group else []
                crossings_links_inds = action_rope_nodes_group['crossings_links_inds'][()] if 'crossings_links_inds' in action_rope_nodes_group else []
                cross_pairs = set([tuple(p) for p in action_rope_nodes_group['cross_pairs']]) if 'cross_pairs' in action_rope_nodes_group else set([])
                rope_closed = action_rope_nodes_group['rope_closed'][()]
            GlobalVars.rope_nodes_crossing_info[action] = (old_rope_nodes, crossings, crossings_links_inds, cross_pairs, rope_closed)
        state_id, new_cloud, new_rope_nodes = state
        if state_id not in GlobalVars.rope_nodes_crossing_info:
            crossings, crossings_links_inds, cross_pairs, rope_closed = calculateCrossings(new_rope_nodes)
            GlobalVars.rope_nodes_crossing_info[state_id] = (new_rope_nodes, crossings, crossings_links_inds, cross_pairs, rope_closed)
        def plot_cb(rope_nodes0, rope_nodes1, corr_nm, f, pts_segmentation_inds0, pts_segmentation_inds1):
            from rapprentice.plotting_plt import plot_tps_registration_segment_proj_2d
            import matplotlib.pyplot as plt
            plot_tps_registration_segment_proj_2d(rope_nodes0, rope_nodes1, corr_nm, f, pts_segmentation_inds0, pts_segmentation_inds1)
            plt.show()
        x_weights = np.ones(len(old_cloud)) * 1.0/len(old_cloud)
        f, corr = tps_registration.tps_segment_registration(GlobalVars.rope_nodes_crossing_info[action], GlobalVars.rope_nodes_crossing_info[state_id], 
                                                            cloud0 = old_cloud, cloud1 = new_cloud, corr_tile_pattern = np.eye(args_eval.upsample_rad), 
                                                            reg=np.array([0.00015, 0.00015, 0.0015]), x_weights=x_weights, plotting=False, plot_cb=plot_cb)
    elif reg_type == 'rpm':
        vis_cost_xy = tps_registration.ab_cost(old_cloud, new_cloud) if args_eval.use_color else None
        f, corr = tps_registration.tps_rpm(old_cloud[:,:3], new_cloud[:,:3], n_iter=14, em_iter=1, vis_cost_xy=vis_cost_xy, user_data={'old_cloud':old_cloud, 'new_cloud':new_cloud})
    elif reg_type == 'bij':
        vis_cost_xy = tps_registration.ab_cost(old_cloud, new_cloud) if args_eval.use_color else None
        x_nd = old_cloud[:,:3]
        y_md = new_cloud[:,:3]
        scaled_x_nd, src_params = registration.unit_boxify(x_nd)
        scaled_y_md, targ_params = registration.unit_boxify(y_md)
        f,g = registration.tps_rpm_bij(scaled_x_nd, scaled_y_md, rot_reg=1e-3, n_iter=10, vis_cost_xy=vis_cost_xy)
        corr = None
    return f, corr

def compute_trans_traj(sim_env, state_or_get_state_fn, action, i_step, args_eval, transferopt=None, animate=False, interactive=False, simulate=True, replay_full_trajs=None):
    alpha = args_eval.alpha
    beta_pos = args_eval.beta_pos
    beta_rot = args_eval.beta_rot
    gamma = args_eval.gamma
    if transferopt is None:
        transferopt = args_eval.transferopt
    
    seg_info = GlobalVars.actions[action]
    if simulate:
        sim_util.reset_arms_to_side(sim_env)
    
    cloud_dim = 6 if args_eval.use_color else 3
    old_cloud = get_action_cloud_ds(sim_env, action, args_eval)[:,:cloud_dim]
    old_rope_nodes = get_action_rope_nodes(sim_env, action, args_eval)
    
    closing_inds = get_closing_inds(seg_info)
    closing_hmats = {}
    for lr in closing_inds:
        if closing_inds[lr] != -1:
            closing_hmats[lr] = seg_info["%s_gripper_tool_frame"%lr]['hmat'][closing_inds[lr]]
    
    miniseg_intervals = []
    for lr in 'lr':
        miniseg_intervals.extend([(i_miniseg_lr, lr, i_start, i_end) for (i_miniseg_lr, (i_start, i_end)) in enumerate(zip(*sim_util.split_trajectory_by_lr_gripper(seg_info, lr)))])
    # sort by the start of the trajectory, then by the length (if both trajectories start at the same time, the shorter one should go first), and then break ties by executing the right trajectory first
    miniseg_intervals = sorted(miniseg_intervals, key=lambda (i_miniseg_lr, lr, i_start, i_end): (i_start, i_end-i_start, {'l':'r', 'r':'l'}[lr]))
    
    miniseg_interval_groups = []
    for (curr_miniseg_interval, next_miniseg_interval) in zip(miniseg_intervals[:-1], miniseg_intervals[1:]):
        curr_i_miniseg_lr, curr_lr, curr_i_start, curr_i_end = curr_miniseg_interval
        next_i_miniseg_lr, next_lr, next_i_start, next_i_end = next_miniseg_interval
        if len(miniseg_interval_groups) > 0 and curr_miniseg_interval in miniseg_interval_groups[-1]:
            continue
        curr_gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%curr_lr][curr_i_end])
        next_gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%next_lr][next_i_end])
        miniseg_interval_group = [curr_miniseg_interval]
        if not curr_gripper_open and not next_gripper_open and curr_lr != next_lr and curr_i_start < next_i_end and next_i_start < curr_i_end:
            miniseg_interval_group.append(next_miniseg_interval)
        miniseg_interval_groups.append(miniseg_interval_group)
    
    success = True
    feasible = True
    misgrasp = False
    full_trajs = []
    obj_values = []
    for i_miniseg_group, miniseg_interval_group in enumerate(miniseg_interval_groups):
        if type(state_or_get_state_fn) == tuple:
            state = state_or_get_state_fn
        else:
            state = state_or_get_state_fn(sim_env)
        if state is None: break
        _, new_cloud, new_rope_nodes = state
        new_cloud = new_cloud[:,:cloud_dim]
    
        handles = []
        if animate:
            # color code: r demo, y transformed, g transformed resampled, b new
            handles.append(sim_env.env.plot3(old_cloud[:,:3], 2, (1,0,0)))
            handles.append(sim_env.env.plot3(new_cloud[:,:3], 2, new_cloud[:,3:] if args_eval.use_color else (0,0,1)))
            sim_env.viewer.Step()

        if not simulate or replay_full_trajs is None: # we are not simulating, we still want to compute the costs
            group_full_trajs = []
            for (i_miniseg_lr, lr, i_start, i_end) in miniseg_interval_group:
                manip_name = {"l":"leftarm", "r":"rightarm"}[lr]                 
                ee_link_name = "%s_gripper_tool_frame"%lr
        
                ################################    
                redprint("Generating %s arm joint trajectory for part %i"%(lr, i_miniseg_lr))
                
                # figure out how we're gonna resample stuff
                old_arm_traj = asarray(seg_info[manip_name][i_start - int(i_start > 0):i_end+1])
                if not sim_util.arm_moved(old_arm_traj):
                    continue
                old_finger_traj = sim_util.gripper_joint2gripper_l_finger_joint_values(seg_info['%s_gripper_joint'%lr][i_start - int(i_start > 0):i_end+1])[:,None]
                JOINT_LENGTH_PER_STEP = .1
                _, timesteps_rs = sim_util.unif_resample(old_arm_traj, JOINT_LENGTH_PER_STEP)
            
                ### Generate fullbody traj
                old_arm_traj_rs = mu.interp2d(timesteps_rs, np.arange(len(old_arm_traj)), old_arm_traj)

                f, corr = register_tps(sim_env, state, action, args_eval)
                if f is None: break

                if animate:
                    handles.append(sim_env.env.plot3(f.transform_points(old_cloud[:,:3]), 2, old_cloud[:,3:] if args_eval.use_color else (1,1,0)))
                    new_cloud_rs = corr.dot(new_cloud)
                    handles.append(sim_env.env.plot3(new_cloud_rs[:,:3], 2, new_cloud_rs[:,3:] if args_eval.use_color else (0,1,0)))
                    handles.extend(draw_grid(sim_env, old_cloud[:,:3], f))
                
                x_na = old_cloud
                y_ng = (corr/corr.sum(axis=1)[:,None]).dot(new_cloud)
                bend_coef = f._bend_coef
                rot_coef = f._rot_coef
                wt_n = f._wt_n.copy()
                
                interest_pts_inds = np.zeros(len(old_cloud), dtype=bool)
                if lr in closing_hmats:
                    interest_pts_inds += np.apply_along_axis(np.linalg.norm, 1, old_cloud - closing_hmats[lr][:3,3]) < 0.05
    
                interest_pts_err_tol = 0.0025
                max_iters = 5 if transferopt != "pose" else 0
                penalty_factor = 10.0
                
                if np.any(interest_pts_inds):
                    for _ in range(max_iters):
                        interest_pts_errs = np.apply_along_axis(np.linalg.norm, 1, (f.transform_points(x_na[interest_pts_inds,:]) - y_ng[interest_pts_inds,:]))
                        if np.all(interest_pts_errs < interest_pts_err_tol):
                            break
                        redprint("TPS fitting: The error of the interest points is above the tolerance. Increasing penalty for these weights.")
                        wt_n[interest_pts_inds] *= penalty_factor
                        f = registration.fit_ThinPlateSpline(x_na, y_ng, bend_coef, rot_coef, wt_n)
        
                old_ee_traj = asarray(seg_info["%s_gripper_tool_frame"%lr]['hmat'][i_start - int(i_start > 0):i_end+1])
                transformed_ee_traj = f.transform_hmats(old_ee_traj)
                transformed_ee_traj_rs = np.asarray(resampling.interp_hmats(timesteps_rs, np.arange(len(transformed_ee_traj)), transformed_ee_traj))
                 
                if animate:
                    handles.append(sim_env.env.drawlinestrip(old_ee_traj[:,:3,3], 2, (1,0,0)))
                    handles.append(sim_env.env.drawlinestrip(transformed_ee_traj[:,:3,3], 2, (1,1,0)))
                    handles.append(sim_env.env.drawlinestrip(transformed_ee_traj_rs[:,:3,3], 2, (0,1,0)))
                    sim_env.viewer.Step()
                
                print "planning pose trajectory following"
                dof_inds = sim_util.dof_inds_from_name(sim_env.robot, manip_name)
                joint_ind = sim_env.robot.GetJointIndex("%s_shoulder_lift_joint"%lr)
                init_arm_traj = old_arm_traj_rs.copy()
                init_arm_traj[:,dof_inds.index(joint_ind)] = sim_env.robot.GetDOFLimits([joint_ind])[0][0]
                new_arm_traj, obj_value, pose_errs = planning.plan_follow_traj(sim_env.robot, manip_name, sim_env.robot.GetLink(ee_link_name), transformed_ee_traj_rs, init_arm_traj, 
                                                                               start_fixed=i_miniseg_lr!=0,
                                                                               beta_pos=beta_pos, beta_rot=beta_rot)
                
                if transferopt == 'finger' or transferopt == 'joint':
                    old_ee_traj_rs = np.asarray(resampling.interp_hmats(timesteps_rs, np.arange(len(old_ee_traj)), old_ee_traj))
                    old_finger_traj_rs = mu.interp2d(timesteps_rs, np.arange(len(old_finger_traj)), old_finger_traj)
                    flr2old_finger_pts_traj_rs = sim_util.get_finger_pts_traj(sim_env, lr, (old_ee_traj_rs, old_finger_traj_rs))
                    
                    flr2transformed_finger_pts_traj_rs = {}
                    flr2finger_link = {}
                    flr2finger_rel_pts = {}
                    for finger_lr in 'lr':
                        flr2transformed_finger_pts_traj_rs[finger_lr] = f.transform_points(np.concatenate(flr2old_finger_pts_traj_rs[finger_lr], axis=0)).reshape((-1,4,3))
                        flr2finger_link[finger_lr] = sim_env.robot.GetLink("%s_gripper_%s_finger_tip_link"%(lr,finger_lr))
                        flr2finger_rel_pts[finger_lr] = sim_util.get_finger_rel_pts(finger_lr)
                    
                    if animate:
                        handles.extend(draw_finger_pts_traj(sim_env, flr2old_finger_pts_traj_rs, (1,0,0)))
                        handles.extend(draw_finger_pts_traj(sim_env, flr2transformed_finger_pts_traj_rs, (0,1,0)))
                        sim_env.viewer.Step()
                        
                    # enable finger DOF and extend the trajectories to include the closing part only if the gripper closes at the end of this minisegment
                    next_gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%lr][i_end+1]) if i_end+1 < len(seg_info["%s_gripper_joint"%lr]) else True
                    if not sim_env.sim.is_grabbing_rope(lr) and not next_gripper_open:
                        manip_name = manip_name + "+" + "%s_gripper_l_finger_joint"%lr
                        
                        old_finger_closing_traj_start = old_finger_traj_rs[-1][0]
                        old_finger_closing_traj_target = sim_util.get_binary_gripper_angle(sim_util.binarize_gripper(seg_info["%s_gripper_joint"%lr][i_end+1]))
                        old_finger_closing_traj_rs = np.linspace(old_finger_closing_traj_start, old_finger_closing_traj_target, np.ceil(abs(old_finger_closing_traj_target - old_finger_closing_traj_start) / .02))[:,None]
                        closing_n_steps = len(old_finger_closing_traj_rs)
                        old_ee_closing_traj_rs = np.tile(old_ee_traj_rs[-1], (closing_n_steps,1,1))
                        flr2old_finger_pts_closing_traj_rs = sim_util.get_finger_pts_traj(sim_env, lr, (old_ee_closing_traj_rs, old_finger_closing_traj_rs))
                          
                        init_traj = np.r_[np.c_[new_arm_traj,                                   old_finger_traj_rs],
                                            np.c_[np.tile(new_arm_traj[-1], (closing_n_steps,1)), old_finger_closing_traj_rs]]
                        flr2transformed_finger_pts_closing_traj_rs = {}
                        for finger_lr in 'lr':
                            flr2old_finger_pts_traj_rs[finger_lr] = np.r_[flr2old_finger_pts_traj_rs[finger_lr], flr2old_finger_pts_closing_traj_rs[finger_lr]]
                            flr2transformed_finger_pts_closing_traj_rs[finger_lr] = f.transform_points(np.concatenate(flr2old_finger_pts_closing_traj_rs[finger_lr], axis=0)).reshape((-1,4,3))
                            flr2transformed_finger_pts_traj_rs[finger_lr] = np.r_[flr2transformed_finger_pts_traj_rs[finger_lr],
                                                                                  flr2transformed_finger_pts_closing_traj_rs[finger_lr]]
                        
                        if animate:
                            handles.extend(draw_finger_pts_traj(sim_env, flr2old_finger_pts_closing_traj_rs, (1,0,0)))
                            handles.extend(draw_finger_pts_traj(sim_env, flr2transformed_finger_pts_closing_traj_rs, (0,1,0)))
                            sim_env.viewer.Step()
                    else:
                        init_traj = new_arm_traj
                    
                    print "planning finger points trajectory following"
                    new_traj, obj_value, pose_errs = planning.plan_follow_finger_pts_traj(sim_env.robot, manip_name, flr2finger_link, flr2finger_rel_pts, flr2transformed_finger_pts_traj_rs, init_traj, 
                                                                                          start_fixed=i_miniseg_lr!=0,
                                                                                          beta_pos=beta_pos, gamma=gamma)
                    
                    if transferopt == 'joint':
                        print "planning joint TPS and finger points trajectory following"
                        new_traj, f, new_N_z, \
                        obj_value, rel_pts_costs, tps_cost = planning.joint_fit_tps_follow_finger_pts_traj(sim_env.robot, manip_name, flr2finger_link, flr2finger_rel_pts, flr2old_finger_pts_traj_rs, new_traj, 
                                                                                                           x_na, y_ng, bend_coef, rot_coef, wt_n, old_N_z=None,
                                                                                                           start_fixed=i_miniseg_lr!=0,
                                                                                                           alpha=alpha, beta_pos=beta_pos, gamma=gamma)
                        if np.any(interest_pts_inds):
                            for _ in range(max_iters):
                                interest_pts_errs = np.apply_along_axis(np.linalg.norm, 1, (f.transform_points(x_na[interest_pts_inds,:]) - y_ng[interest_pts_inds,:]))
                                if np.all(interest_pts_errs < interest_pts_err_tol):
                                    break
                                redprint("Joint TPS fitting: The error of the interest points is above the tolerance. Increasing penalty for these weights.")
                                wt_n[interest_pts_inds] *= penalty_factor
                                new_traj, f, new_N_z, \
                                obj_value, rel_pts_costs, tps_cost = planning.joint_fit_tps_follow_finger_pts_traj(sim_env.robot, manip_name, flr2finger_link, flr2finger_rel_pts, flr2old_finger_pts_traj_rs, new_traj, 
                                                                                                                   x_na, y_ng, bend_coef, rot_coef, wt_n, old_N_z=new_N_z,
                                                                                                                   start_fixed=i_miniseg_lr!=0,
                                                                                                                   alpha=alpha, beta_pos=beta_pos, gamma=gamma)
                    else:
                        obj_value += alpha * planning.tps_obj(f, x_na, y_ng, bend_coef, rot_coef, wt_n)
                    
                    if animate:
                        flr2new_transformed_finger_pts_traj_rs = {}
                        for finger_lr in 'lr':
                            flr2new_transformed_finger_pts_traj_rs[finger_lr] = f.transform_points(np.concatenate(flr2old_finger_pts_traj_rs[finger_lr], axis=0)).reshape((-1,4,3))
                        handles.extend(draw_finger_pts_traj(sim_env, flr2new_transformed_finger_pts_traj_rs, (0,1,1)))
                        sim_env.viewer.Step()
                else:
                    new_traj = new_arm_traj
                
                obj_values.append(obj_value)
                
                f._bend_coef = bend_coef
                f._rot_coef = rot_coef
                f._wt_n = wt_n
                
                full_traj = (new_traj, sim_util.dof_inds_from_name(sim_env.robot, manip_name))
                group_full_trajs.append(full_traj)
    
                if animate:
                    handles.append(sim_env.env.drawlinestrip(sim_util.get_ee_traj(sim_env, lr, full_traj)[:,:3,3], 2, (0,0,1)))
                    flr2new_finger_pts_traj = sim_util.get_finger_pts_traj(sim_env, lr, full_traj)
                    handles.extend(draw_finger_pts_traj(sim_env, flr2new_finger_pts_traj, (0,0,1)))
                    sim_env.viewer.Step()
            full_traj = sim_util.merge_full_trajs(group_full_trajs)
        else:
            full_traj = replay_full_trajs[i_miniseg_group]
        full_trajs.append(full_traj)
        
        if not simulate:
            if not eval_util.traj_is_safe(sim_env, full_traj, COLLISION_DIST_THRESHOLD, upsample=100):
                return np.inf
            else:
                continue

        for (i_miniseg_lr, lr, _, _) in miniseg_interval_group:
            redprint("Executing %s arm joint trajectory for part %i"%(lr, i_miniseg_lr))
        
        if len(full_traj[0]) > 0:
            if not eval_util.traj_is_safe(sim_env, full_traj, COLLISION_DIST_THRESHOLD, upsample=100):
                redprint("Trajectory not feasible")
                feasible = False
                success = False
            else:  # Only execute feasible trajectories
                first_miniseg = True
                for (i_miniseg_lr, _, _, _) in miniseg_interval_group:
                    first_miniseg &= i_miniseg_lr == 0
                if len(full_traj[0]) > 0:
                    success &= sim_util.sim_full_traj_maybesim(sim_env, full_traj, animate=animate, interactive=interactive, max_cart_vel_trans_traj=.05 if first_miniseg else .02)

        if not success: break
        
        for (i_miniseg_lr, lr, i_start, i_end) in miniseg_interval_group:
            next_gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%lr][i_end+1]) if i_end+1 < len(seg_info["%s_gripper_joint"%lr]) else True
            curr_gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%lr][i_end])
            if not sim_util.set_gripper_maybesim(sim_env, lr, next_gripper_open, curr_gripper_open, animate=animate):
                redprint("Grab %s failed" % lr)
                misgrasp = True
                success = False

        if not success: break

    if not simulate:
        return np.sum(obj_values)

    sim_env.sim.settle(animate=animate)
    sim_env.sim.release_rope('l')
    sim_env.sim.release_rope('r')
    sim_util.reset_arms_to_side(sim_env)
    if animate:
        sim_env.viewer.Step()
    
    return success, feasible, misgrasp, full_trajs

def regcost_feature_fn(sim_env, state, action, args_eval):
    f, corr = register_tps_cheap(sim_env, state, action, args_eval)
    if f is None:
        cost = np.inf
    else:
        cost = registration.tps_reg_cost(f)
    return np.array([float(cost)]) # no need to normalize since bending cost is independent of number of points

def regcost_trajopt_feature_fn(sim_env, state, action, args_eval):
    obj_values_sum = compute_trans_traj(sim_env, state, action, None, args_eval, simulate=False, transferopt='finger')
    return np.array([obj_values_sum])

def jointopt_feature_fn(sim_env, state, action, args_eval):
    obj_values_sum = compute_trans_traj(sim_env, state, action, None, args_eval, simulate=False, transferopt='joint')
    return np.array([obj_values_sum])

def q_value_fn(state, action, sim_env, fn):
    return np.dot(WEIGHTS, fn(sim_env, state, action)) #+ w0

def select_feature_fn(warping_cost, args_eval):
    if warping_cost == "regcost":
        feature_fn = lambda sim_env, state, action: regcost_feature_fn(sim_env, state, action, args_eval)
    elif warping_cost == "regcost-trajopt":
        feature_fn = lambda sim_env, state, action: regcost_trajopt_feature_fn(sim_env, state, action, args_eval)
    elif warping_cost == "jointopt":
        feature_fn = lambda sim_env, state, action: jointopt_feature_fn(sim_env, state, action, args_eval)
    else:
        raise RuntimeError("Invalid warping cost")
    return feature_fn

def get_unique_id(): 
    GlobalVars.unique_id += 1
    return GlobalVars.unique_id - 1

def get_state(sim_env, args_eval):
    if args_eval.raycast:
        new_cloud, endpoint_inds = sim_env.sim.raycast_cloud(endpoints=3)
        if new_cloud.shape[0] == 0: # rope is not visible (probably because it fall off the table)
            return None
    else:
        new_cloud = sim_env.sim.observe_cloud(upsample=args_eval.upsample, upsample_rad=args_eval.upsample_rad)
        endpoint_inds = np.zeros(len(new_cloud), dtype=bool) # for now, args_eval.raycast=False is not compatible with args_eval.use_color=True
    if args_eval.use_color:
        new_cloud = color_cloud(new_cloud, endpoint_inds)
    new_cloud_ds = clouds.downsample(new_cloud, DS_SIZE) if args_eval.downsample else new_cloud
    new_rope_nodes = sim_env.sim.rope.GetControlPoints()
    new_rope_nodes= ropesim.observe_cloud(new_rope_nodes, sim_env.sim.rope_params.radius, upsample=args_eval.upsample)
    state = ("eval_%i"%get_unique_id(), new_cloud_ds, new_rope_nodes)
    return state

def select_best_action(sim_env, state, num_actions_to_try, feature_fn, prune_feature_fn, eval_stats, warpingcost):
    print "Choosing an action"
    num_top_actions = max(num_actions_to_try, TRAJOPT_MAX_ACTIONS)

    start_time = time.time()
    q_values_prune = [(q_value_fn(state, action, sim_env, prune_feature_fn), action) for action in GlobalVars.actions]
    agenda_top_actions = sorted(q_values_prune, key = lambda v: -v[0])[:num_top_actions]

    if feature_fn == prune_feature_fn:
        q_values = q_values_prune
        agenda = agenda_top_actions
    else:
        if len(agenda_top_actions) > TRAJOPT_MAX_ACTIONS:
            agenda_top_actions = agenda_top_actions[:TRAJOPT_MAX_ACTIONS]
        q_values = [(q_value_fn(state, a, sim_env, feature_fn), a) for (v, a) in agenda_top_actions]
        agenda = sorted(q_values, key = lambda v: -v[0])

    eval_stats.action_elapsed_time += time.time() - start_time
    q_values_root = [q for (q, a) in q_values]
    return agenda, q_values_root

def eval_on_holdout(args, sim_env):
    feature_fn = select_feature_fn(args.eval.warpingcost, args.eval)
    if args.eval.warpingcost == 'regcost':
        prune_feature_fn = feature_fn # so that we can check for function equality
    else:
        prune_feature_fn = select_feature_fn('regcost', args.eval)
    holdoutfile = h5py.File(args.eval.holdoutfile, 'r')
    holdout_items = eval_util.get_holdout_items(holdoutfile, args.tasks, args.taskfile, args.i_start, args.i_end)

    num_successes = 0
    num_total = 0

    for i_task, demo_id_rope_nodes in holdout_items:
        redprint("task %s" % i_task)
        sim_util.reset_arms_to_side(sim_env)
        init_rope_nodes = demo_id_rope_nodes["rope_nodes"][:]
        # don't call replace_rope and sim.settle() directly. use time machine interface for deterministic results!
        time_machine = sim_util.RopeSimTimeMachine(init_rope_nodes, sim_env)

        if args.animation:
            sim_env.viewer.Step()

        eval_util.save_task_results_init(args.resultfile, i_task, sim_env, init_rope_nodes)

        for i_step in range(args.eval.num_steps):
            redprint("task %s step %i" % (i_task, i_step))
            sim_util.reset_arms_to_side(sim_env)
            
            get_state_fn = lambda sim_env: get_state(sim_env, args.eval)
            state = get_state_fn(sim_env)
            _, new_cloud_ds, new_rope_nodes = get_state_fn(sim_env)

            num_actions_to_try = MAX_ACTIONS_TO_TRY if args.eval.search_until_feasible else 1
            eval_stats = eval_util.EvalStats()

            agenda, q_values_root = select_best_action(sim_env, state, num_actions_to_try, feature_fn, prune_feature_fn, eval_stats, args.eval.warpingcost)

            time_machine.set_checkpoint('prechoice_%i'%i_step, sim_env)
            for i_choice in range(num_actions_to_try):
                if agenda[i_choice][0] == -np.inf: # none of the demonstrations generalize
                    break
                redprint("TRYING %s"%agenda[i_choice][1])

                time_machine.restore_from_checkpoint('prechoice_%i'%i_step, sim_env, sim_util.get_rope_params(args.eval.rope_params))
                best_root_action = agenda[i_choice][1]
                start_time = time.time()
                pre_trans, pre_rots = sim_util.get_rope_transforms(sim_env)
                eval_stats.success, eval_stats.feasible, eval_stats.misgrasp, full_trajs = compute_trans_traj(sim_env, get_state_fn, best_root_action, i_step, args.eval, animate=args.animation, interactive=args.interactive)
                trans, rots = sim_util.get_rope_transforms(sim_env)
                eval_stats.exec_elapsed_time += time.time() - start_time

                if eval_stats.feasible:  # try next action if TrajOpt cannot find feasible action
                     eval_stats.found_feasible_action = True
                     break
                else:
                     redprint('TRYING NEXT ACTION')

            if not eval_stats.feasible:  # If not feasible, restore_from_checkpoint
                time_machine.restore_from_checkpoint('prechoice_%i'%i_step, sim_env, sim_util.get_rope_params(args.eval.rope_params))
            print "BEST ACTION:", best_root_action
            
            eval_util.save_task_results_step(args.resultfile, i_task, i_step, sim_env, best_root_action, q_values_root, full_trajs, eval_stats, new_cloud_ds=new_cloud_ds, new_rope_nodes=new_rope_nodes)
            
            if not eval_stats.found_feasible_action:
                # Skip to next knot tie if the action is infeasible -- since
                # that means all future steps (up to 5) will have infeasible trajectories
                break
            
            if is_knot(sim_env.sim.rope.GetControlPoints()):
                break;

        if is_knot(sim_env.sim.rope.GetControlPoints()):
            num_successes += 1
        num_total += 1

        redprint('Eval Successes / Total: ' + str(num_successes) + '/' + str(num_total))

def replay_on_holdout(args, sim_env):
    holdoutfile = h5py.File(args.eval.holdoutfile, 'r')
    loadresultfile = h5py.File(args.replay.loadresultfile, 'r')
    loadresult_items = eval_util.get_holdout_items(loadresultfile, args.tasks, args.taskfile, args.i_start, args.i_end)

    num_successes = 0
    num_total = 0
    
    for i_task, _ in loadresult_items:
        redprint("task %s" % i_task)
        sim_util.reset_arms_to_side(sim_env)
        _, _, _, init_rope_nodes = eval_util.load_task_results_init(args.replay.loadresultfile, i_task)
        # don't call replace_rope and sim.settle() directly. use time machine interface for deterministic results!
        time_machine = sim_util.RopeSimTimeMachine(init_rope_nodes, sim_env)

        if args.animation:
            sim_env.viewer.Step()

        eval_util.save_task_results_init(args.resultfile, i_task, sim_env, init_rope_nodes)
        
        for i_step in range(len(loadresultfile[i_task]) - (1 if 'init' in loadresultfile[i_task] else 0)):
            if args.replay.simulate_traj_steps is not None and i_step not in args.replay.simulate_traj_steps:
                continue
            
            redprint("task %s step %i" % (i_task, i_step))
            sim_util.reset_arms_to_side(sim_env)

            restore_from_saved_trans_rots = args.replay.simulate_traj_steps is not None
            if restore_from_saved_trans_rots:
                if i_step == 0:
                    pre_trans, pre_rots = eval_util.load_task_results_init(args.replay.loadresultfile, i_task)[:2]
                else:
                    pre_trans, pre_rots = eval_util.load_task_results_step(args.replay.loadresultfile, i_task, i_step-1)[:2]
                time_machine.set_checkpoint('preexec_%i'%i_step, sim_env, tfs=(pre_trans, pre_rots))
            else:
                time_machine.set_checkpoint('preexec_%i'%i_step, sim_env)
            time_machine.restore_from_checkpoint('preexec_%i'%i_step, sim_env, sim_util.get_rope_params(args.eval.rope_params))
            
            get_state_fn = lambda sim_env: get_state(sim_env, args.eval)
            _, new_cloud_ds, new_rope_nodes = get_state_fn(sim_env)

            eval_stats = eval_util.EvalStats()

            trans, rots, _, best_action, q_values, replay_full_trajs, _, _ = eval_util.load_task_results_step(args.replay.loadresultfile, i_task, i_step)
            
            if q_values.max() == -np.inf: # none of the demonstrations generalize
                break
            
            start_time = time.time()
            if i_step in args.replay.compute_traj_steps: # compute the trajectory in this step
                replay_full_trajs = None
            eval_stats.success, eval_stats.feasible, eval_stats.misgrasp, full_trajs = compute_trans_traj(sim_env, get_state_fn, best_action, i_step, args.eval, animate=args.animation, interactive=args.interactive, replay_full_trajs=replay_full_trajs)
            eval_stats.exec_elapsed_time += time.time() - start_time

            if eval_stats.feasible:
                 eval_stats.found_feasible_action = True

            if not eval_stats.feasible:  # If not feasible, restore_from_checkpoint
                time_machine.restore_from_checkpoint('preexec_%i'%i_step, sim_env, sim_util.get_rope_params(args.eval.rope_params))
            print "BEST ACTION:", best_action

            replay_trans, replay_rots = sim_util.get_rope_transforms(sim_env)
            if np.all(trans == replay_trans) and np.all(rots == replay_rots):
                yellowprint("Reproducible results OK")
            else:
                yellowprint("The rope transforms of the replay rope doesn't match the ones in the original result file by %f and %f" % (np.linalg.norm(trans - replay_trans), np.linalg.norm(rots - replay_rots)))
            
            eval_util.save_task_results_step(args.resultfile, i_task, i_step, sim_env, best_action, q_values, full_trajs, eval_stats)
            
            if not eval_stats.found_feasible_action:
                # Skip to next knot tie if the action is infeasible -- since
                # that means all future steps (up to 5) will have infeasible trajectories
                break
            
            if is_knot(sim_env.sim.rope.GetControlPoints()):
                break;

        if is_knot(sim_env.sim.rope.GetControlPoints()):
            num_successes += 1
        num_total += 1

        redprint('REPLAY Successes / Total: ' + str(num_successes) + '/' + str(num_total))

def parse_input_args():
    parser = util.ArgumentParser()
    
    parser.add_argument("--animation", type=int, default=0, help="if greater than 1, the viewer tries to load the window and camera properties without idling at the beginning")
    parser.add_argument("--interactive", action="store_true", help="step animation and optimization if specified")
    parser.add_argument("--resultfile", type=str, help="no results are saved if this is not specified")

    # selects tasks to evaluate/replay
    parser.add_argument("--tasks", type=int, nargs='*', metavar="i_task")
    parser.add_argument("--taskfile", type=str)
    parser.add_argument("--i_start", type=int, default=-1, metavar="i_task")
    parser.add_argument("--i_end", type=int, default=-1, metavar="i_task")
    
    parser.add_argument("--camera_matrix_file", type=str, default='.camera_matrix.txt')
    parser.add_argument("--window_prop_file", type=str, default='.win_prop.txt')
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--log", type=str, default="")
    parser.add_argument("--print_mean_and_var", action="store_true")

    subparsers = parser.add_subparsers(dest='subparser_name')

    parser_eval = subparsers.add_parser('eval')
    
    parser_eval.add_argument('actionfile', type=str, nargs='?', default='data/misc/actions.h5')
    parser_eval.add_argument('holdoutfile', type=str, nargs='?', default='data/misc/holdout_set.h5')

    parser_eval.add_argument('warpingcost', type=str, choices=['regcost', 'regcost-trajopt', 'jointopt'])
    parser_eval.add_argument("transferopt", type=str, choices=['pose', 'finger', 'joint'])
    
    parser_eval.add_argument("--obstacles", type=str, nargs='*', choices=['bookshelve', 'boxes', 'cylinders'], default=[])
    parser_eval.add_argument("--raycast", type=int, default=0, help="use raycast or rope nodes observation model")
    parser_eval.add_argument("--downsample", type=int, default=1)
    parser_eval.add_argument("--upsample", type=int, default=0)
    parser_eval.add_argument("--upsample_rad", type=int, default=1, help="upsample_rad > 1 incompatible with downsample != 0")
    
    parser_eval.add_argument("--fake_data_segment",type=str, default='demo1-seg00')
    parser_eval.add_argument("--fake_data_transform", type=float, nargs=6, metavar=("tx","ty","tz","rx","ry","rz"),
        default=[0,0,0,0,0,0], help="translation=(tx,ty,tz), axis-angle rotation=(rx,ry,rz)")
    
    parser_eval.add_argument("--search_until_feasible", action="store_true")
    parser_eval.add_argument("--alpha", type=float, default=10000000.0)
    parser_eval.add_argument("--beta_pos", type=float, default=10000.0)
    parser_eval.add_argument("--beta_rot", type=float, default=10.0)
    parser_eval.add_argument("--gamma", type=float, default=1000.0)
    parser_eval.add_argument("--num_steps", type=int, default=5, help="maximum number of steps to simulate each task")
    parser_eval.add_argument("--use_color", type=int, default=0)
    parser_eval.add_argument("--dof_limits_factor", type=float, default=1.0)
    parser_eval.add_argument("--rope_params", type=str, default='default')

    parser_replay = subparsers.add_parser('replay')
    parser_replay.add_argument("loadresultfile", type=str)
    parser_replay.add_argument("--compute_traj_steps", type=int, default=[], nargs='*', metavar='i_step', help="recompute trajectories for the i_step of all tasks")
    parser_replay.add_argument("--simulate_traj_steps", type=int, default=None, nargs='*', metavar='i_step', 
                               help="if specified, restore the rope state from file and then simulate for the i_step of all tasks")
                               # if not specified, the rope state is not restored from file, but it is as given by the sequential simulation

    return parser.parse_args()

def setup_log_file(args):
    if args.log:
        redprint("Writing log to file %s" % args.log)
        GlobalVars.exec_log = task_execution.ExecutionLog(args.log)
        atexit.register(GlobalVars.exec_log.close)
        GlobalVars.exec_log(0, "main.args", args)

def set_global_vars(args, sim_env):
    if args.random_seed is not None: np.random.seed(args.random_seed)

    GlobalVars.actions = h5py.File(args.eval.actionfile, 'r')
    actions_root, actions_ext = os.path.splitext(args.eval.actionfile)
    GlobalVars.actions_cache = h5py.File(actions_root + '.cache' + actions_ext, 'a')
    
    if args.eval.downsample:
        global clouds
        from rapprentice import clouds
    
def load_simulation(args, sim_env):
    sim_env.env = openravepy.Environment()
    sim_env.env.StopSimulation()
#     sim_env.env.Load("robots/pr2-beta-static.zae")
    sim_env.env.Load("./data/misc/pr2-beta-static-decomposed-shoulder.zae")
    sim_env.robot = sim_env.env.GetRobots()[0]

    actions = h5py.File(args.eval.actionfile, 'r')
    
    init_rope_xyz, _ = sim_util.load_fake_data_segment(sim_env, actions, args.eval.fake_data_segment, args.eval.fake_data_transform) # this also sets the torso (torso_lift_joint) to the height in the data
    table_height = init_rope_xyz[:,2].mean() - .02
    table_xml = sim_util.make_table_xml(translation=[1, 0, table_height + (-.1 + .01)], extents=[.85, .85, .1])
#     table_xml = sim_util.make_table_xml(translation=[1-.3, 0, table_height + (-.1 + .01)], extents=[.85-.3, .85-.3, .1])
    sim_env.env.LoadData(table_xml)
    obstacle_bodies = []
    if 'bookshelve' in args.eval.obstacles:
        sim_env.env.Load("data/bookshelves.env.xml")
        obstacle_bodies.extend(sim_env.env.GetBodies()[-1:])
    if 'boxes' in args.eval.obstacles:
        sim_env.env.LoadData(sim_util.make_box_xml("box0", [.7,.43,table_height+(.01+.12)], [.12,.12,.12]))
        sim_env.env.LoadData(sim_util.make_box_xml("box1", [.74,.47,table_height+(.01+.12*2+.08)], [.08,.08,.08]))
        obstacle_bodies.extend(sim_env.env.GetBodies()[-2:])
    if 'cylinders' in args.eval.obstacles:
        sim_env.env.LoadData(sim_util.make_cylinder_xml("cylinder0", [.7,.43,table_height+(.01+.5)], .12, 1.))
        sim_env.env.LoadData(sim_util.make_cylinder_xml("cylinder1", [.7,-.43,table_height+(.01+.5)], .12, 1.))
        sim_env.env.LoadData(sim_util.make_cylinder_xml("cylinder2", [.4,.2,table_height+(.01+.65)], .06, .5))
        sim_env.env.LoadData(sim_util.make_cylinder_xml("cylinder3", [.4,-.2,table_height+(.01+.65)], .06, .5))
        obstacle_bodies.extend(sim_env.env.GetBodies()[-4:])

    cc = trajoptpy.GetCollisionChecker(sim_env.env)
    for gripper_link in [link for link in sim_env.robot.GetLinks() if 'gripper' in link.GetName()]:
        cc.ExcludeCollisionPair(gripper_link, sim_env.env.GetKinBody('table').GetLinks()[0])

    sim_util.reset_arms_to_side(sim_env)
    
    if args.animation:
        sim_env.viewer = trajoptpy.GetViewer(sim_env.env)
        if args.animation > 1 and os.path.isfile(args.window_prop_file) and os.path.isfile(args.camera_matrix_file):
            print "loading window and camera properties"
            window_prop = np.loadtxt(args.window_prop_file)
            camera_matrix = np.loadtxt(args.camera_matrix_file)
            try:
                sim_env.viewer.SetWindowProp(*window_prop)
                sim_env.viewer.SetCameraManipulatorMatrix(camera_matrix)
            except:
                print "SetWindowProp and SetCameraManipulatorMatrix are not defined. Pull and recompile Trajopt."
        else:
            print "move viewer to viewpoint that isn't stupid"
            print "then hit 'p' to continue"
            sim_env.viewer.Idle()
            print "saving window and camera properties"
            try:
                window_prop = sim_env.viewer.GetWindowProp()
                camera_matrix = sim_env.viewer.GetCameraManipulatorMatrix()
                np.savetxt(args.window_prop_file, window_prop, fmt='%d')
                np.savetxt(args.camera_matrix_file, camera_matrix)
            except:
                print "GetWindowProp and GetCameraManipulatorMatrix are not defined. Pull and recompile Trajopt."
        for body in obstacle_bodies:
            sim_env.viewer.SetTransparency(body, .35)
    
    if args.eval.dof_limits_factor != 1.0:
        assert 0 < args.eval.dof_limits_factor and args.eval.dof_limits_factor <= 1.0
        active_dof_indices = sim_env.robot.GetActiveDOFIndices()
        active_dof_limits = sim_env.robot.GetActiveDOFLimits()
        for lr in 'lr':
            manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
            dof_inds = sim_env.robot.GetManipulator(manip_name).GetArmIndices()
            limits = np.asarray(sim_env.robot.GetDOFLimits(dof_inds))
            limits_mean = limits.mean(axis=0)
            limits_width = np.diff(limits, axis=0)
            new_limits = limits_mean + args.eval.dof_limits_factor * np.r_[-limits_width/2.0, limits_width/2.0]
            for i, ind in enumerate(dof_inds):
                active_dof_limits[0][active_dof_indices.tolist().index(ind)] = new_limits[0,i]
                active_dof_limits[1][active_dof_indices.tolist().index(ind)] = new_limits[1,i]
        sim_env.robot.SetDOFLimits(active_dof_limits[0], active_dof_limits[1])

def main():
    args = parse_input_args()

    if args.subparser_name == "eval":
        eval_util.save_results_args(args.resultfile, args)
    elif args.subparser_name == "replay":
        loaded_args = eval_util.load_results_args(args.replay.loadresultfile)
        assert 'eval' not in vars(args)
        args.eval = loaded_args.eval
    else:
        raise RuntimeError("Invalid subparser name")
    
    setup_log_file(args)
    
    sim_env = sim_util.SimulationEnv()
    set_global_vars(args, sim_env)
    trajoptpy.SetInteractive(args.interactive)
    load_simulation(args, sim_env)

    if args.subparser_name == "eval":
        eval_on_holdout(args, sim_env)
    elif args.subparser_name == "replay":
        replay_on_holdout(args, sim_env)
    else:
        raise RuntimeError("Invalid subparser name")

    if args.print_mean_and_var:
        if GlobalVars.tps_errors_top10:
            print "TPS error mean:", np.mean(GlobalVars.tps_errors_top10)
            print "TPS error variance:", np.var(GlobalVars.tps_errors_top10)
            print "Total Num TPS errors:", len(GlobalVars.tps_errors_top10)
        if GlobalVars.trajopt_errors_top10:
            print "TrajOpt error mean:", np.mean(GlobalVars.trajopt_errors_top10)
            print "TrajOpt error variance:", np.var(GlobalVars.trajopt_errors_top10)
            print "Total Num TrajOpt errors:", len(GlobalVars.trajopt_errors_top10)

if __name__ == "__main__":
    main()
