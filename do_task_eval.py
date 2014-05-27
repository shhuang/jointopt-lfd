#!/usr/bin/env python

from __future__ import division

import pprint
import argparse
import planning, eval_util, sim_util, util
import tps_registration as tps_registration_old
from rapprentice import tps_registration
 
from rapprentice import registration, colorize, berkeley_pr2, \
     animate_traj, ros2rave, plotting_openrave, task_execution, \
     tps, func_utils, resampling, ropesim, rope_initialization, clouds
from rapprentice import math_utils as mu
from rapprentice.yes_or_no import yes_or_no
import pdb, time
 
import cloudprocpy, trajoptpy, openravepy
import rope_qlearn
from knot_classifier import isKnot as is_knot
import os, os.path, numpy as np, h5py
from numpy import asarray
import atexit
import importlib
from itertools import combinations
import IPython as ipy
import random

COLLISION_DIST_THRESHOLD = 0.0
MAX_ACTIONS_TO_TRY = 5  # Number of actions to try (ranked by cost), if TrajOpt trajectory is infeasible
TRAJOPT_MAX_ACTIONS = 10  # Number of actions to compute full feature (TPS + TrajOpt) on
WEIGHTS = np.array([-1]) 
DS_SIZE = .025

class GlobalVars:
    unique_id = 0
    alpha = 20.0  # alpha and beta can be set by user parameters - but should be changed nowhere else
    beta = 10.0
    resample_rope = None
    actions = None
    gripper_weighting = False
    tps_errors_top10 = []
    trajopt_errors_top10 = []
    actions_ds_clouds = {}

def get_ds_cloud(sim_env, action):
    if action not in GlobalVars.actions_ds_clouds:
        GlobalVars.actions_ds_clouds[action] = clouds.downsample(GlobalVars.actions[action]['cloud_xyz'], DS_SIZE)
    return GlobalVars.actions_ds_clouds[action]

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

def compute_trans_traj(sim_env, new_cloud, action, use_color, animate=False, interactive=False, simulate=True):
    seg_info = GlobalVars.actions[action]
    if simulate:
        sim_util.reset_arms_to_side(sim_env)
    
    redprint("Generating end-effector trajectory")    
    
    cloud_dim = 6 if use_color else 3
    new_cloud = new_cloud[:,:cloud_dim]
    old_cloud = get_ds_cloud(sim_env, action)[:,:cloud_dim]
    
    link_names = ["%s_gripper_tool_frame"%lr for lr in ('lr')]
    hmat_list = [(lr, seg_info[ln]['hmat']) for lr, ln in zip('lr', link_names)]
    if GlobalVars.gripper_weighting:
        interest_pts = get_closing_pts(seg_info)
        raise NonImplementedError
    else:
        interest_pts = None
    f, corr = tps_registration.tps_rpm(old_cloud[:,:3], new_cloud[:,:3], vis_cost_xy = tps_registration.ab_cost(old_cloud, new_cloud) if use_color else None, user_data={'old_cloud':old_cloud, 'new_cloud':new_cloud})
    lr2eetraj = {}
    for k, hmats in hmat_list:
        lr2eetraj[k] = f.transform_hmats(hmats)

    handles = []
    if animate:
        handles.append(sim_env.env.plot3(old_cloud[:,:3], 5, (1,0,0)))
        handles.append(sim_env.env.plot3(new_cloud[:,:3], 5, new_cloud[:,3:] if use_color else (0,0,1)))
        handles.append(sim_env.env.plot3(f.transform_points(old_cloud[:,:3]), 5, old_cloud[:,3:] if use_color else (0,1,0)))
        handles.extend(draw_grid(sim_env, old_cloud[:,:3], f))
        for k, old_hmats in hmat_list:
            handles.append(sim_env.env.drawlinestrip(old_hmats[:,:3,3], 2, (1,0,0,1)))
        for transformed_hmats in lr2eetraj.values():
            handles.append(sim_env.env.drawlinestrip(transformed_hmats[:,:3,3], 2, (0,1,0,1)))
        sim_env.viewer.Step()

    miniseg_starts, miniseg_ends = sim_util.split_trajectory_by_gripper(seg_info)    
    success = True
    feasible = True
    misgrasp = False
    print colorize.colorize("mini segments:", "red"), miniseg_starts, miniseg_ends
    full_trajs = []

    total_pose_cost = 0
    total_poses = 0
    for (i_miniseg, (i_start, i_end)) in enumerate(zip(miniseg_starts, miniseg_ends)):            

        ################################    
        redprint("Generating joint trajectory for part %i"%(i_miniseg))

        # figure out how we're gonna resample stuff
        lr2oldtraj = {}
        for lr in 'lr':
            manip_name = {"l":"leftarm", "r":"rightarm"}[lr]                 
            old_joint_traj = asarray(seg_info[manip_name][i_start:i_end+1])
            #print (old_joint_traj[1:] - old_joint_traj[:-1]).ptp(axis=0), i_start, i_end
            if sim_util.arm_moved(old_joint_traj):       
                lr2oldtraj[lr] = old_joint_traj   
        if len(lr2oldtraj) > 0:
            old_total_traj = np.concatenate(lr2oldtraj.values(), 1)
            JOINT_LENGTH_PER_STEP = .1
            _, timesteps_rs = sim_util.unif_resample(old_total_traj, JOINT_LENGTH_PER_STEP)
        ####

        ### Generate fullbody traj
        lr2newtraj = {}

        for (lr,old_joint_traj) in lr2oldtraj.items():
            
            manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
            
            old_joint_traj_rs = mu.interp2d(timesteps_rs, np.arange(len(old_joint_traj)), old_joint_traj)
            
            ee_link_name = "%s_gripper_tool_frame"%lr
            new_ee_traj = lr2eetraj[lr][i_start:i_end+1]          
            new_ee_traj_rs = resampling.interp_hmats(timesteps_rs, np.arange(len(new_ee_traj)), new_ee_traj)
            print "planning trajectory following"
            new_joint_traj, pose_errs = planning.plan_follow_traj(sim_env.robot, manip_name,
                                                       sim_env.robot.GetLink(ee_link_name), new_ee_traj_rs,old_joint_traj_rs, beta=GlobalVars.beta)
            if animate:
                handles.append(sim_env.env.drawlinestrip(sim_util.get_ee_traj(sim_env, lr, new_joint_traj)[:,:3,3], 2, (0,0,1,1)))
                sim_env.viewer.Step()
            
            n_steps = len(new_ee_traj_rs)
            total_pose_cost += pose_errs * float(n_steps) / float(GlobalVars.beta)   # Normalize pos error
            total_poses += n_steps

            lr2newtraj[lr] = new_joint_traj
            ################################    
        redprint("Executing joint trajectory for part %i using arms '%s'"%(i_miniseg, lr2newtraj.keys()))
        full_traj = sim_util.get_full_traj(sim_env, lr2newtraj)
        full_trajs.append(full_traj)
        if not simulate:
            continue

        for lr in 'lr':
            gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%lr][i_start])
            prev_gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%lr][i_start-1]) if i_start != 0 else False
            if not sim_util.set_gripper_maybesim(sim_env, lr, gripper_open, prev_gripper_open):
                redprint("Grab %s failed" % lr)
                misgrasp = True
                success = False

        if not success: break

        if len(full_traj[0]) > 0:
            if not eval_util.traj_is_safe(sim_env, full_traj, COLLISION_DIST_THRESHOLD):
                redprint("Trajectory not feasible")
                feasible = False
                success = False
            else:  # Only execute feasible trajectories
                success &= sim_util.sim_full_traj_maybesim(sim_env, full_traj, animate=animate, interactive=interactive)

        if not success: break

    if not simulate:
        return total_pose_cost / float(total_poses)

    sim_env.sim.settle(animate=animate)
    sim_env.sim.release_rope('l')
    sim_env.sim.release_rope('r')
    sim_util.reset_arms_to_side(sim_env)
    if animate:
        sim_env.viewer.Step()
    
    return success, feasible, misgrasp, full_trajs

def compute_trans_traj_jointopt(sim_env, new_cloud, action, use_color, animate=False, interactive=False, simulate=True):
    seg_info = GlobalVars.actions[action]
    if simulate:
        sim_util.reset_arms_to_side(sim_env)
    
    redprint("Generating end-effector trajectory")    
    
    cloud_dim = 6 if use_color else 3
    new_cloud = new_cloud[:,:cloud_dim]
    old_cloud = get_ds_cloud(sim_env, action)[:,:cloud_dim]

    link_names = ["%s_gripper_tool_frame"%lr for lr in ('lr')]
    hmat_list = [(lr, seg_info[ln]['hmat']) for lr, ln in zip('lr', link_names)]
    hmat_dict = dict(hmat_list)
    if GlobalVars.gripper_weighting:
        interest_pts = get_closing_pts(seg_info)
        raise NonImplementedError
    else:
        interest_pts = None
    f, corr = tps_registration.tps_rpm(old_cloud[:,:3], new_cloud[:,:3], vis_cost_xy = tps_registration.ab_cost(old_cloud, new_cloud) if use_color else None, user_data={'old_cloud':old_cloud, 'new_cloud':new_cloud})
    lr2eetraj = {}
    for k, hmats in hmat_list:
        lr2eetraj[k] = f.transform_hmats(hmats)

    handles = []
    handles_f = [] # handles that depend on the transformation f
    if animate:
        handles.append(sim_env.env.plot3(old_cloud[:,:3], 5, (1,0,0)))
        handles.append(sim_env.env.plot3(new_cloud[:,:3], 5, new_cloud[:,3:] if use_color else (0,0,1)))
        handles_f.append(sim_env.env.plot3(f.transform_points(old_cloud[:,:3]), 5, old_cloud[:,3:] if use_color else (0,1,0)))
        handles_f.extend(draw_grid(sim_env, old_cloud[:,:3], f))
        for k, old_hmats in hmat_list:
            handles.append(sim_env.env.drawlinestrip(old_hmats[:,:3,3], 2, (1,0,0,1)))
        for transformed_hmats in lr2eetraj.values():
            handles_f.append(sim_env.env.drawlinestrip(transformed_hmats[:,:3,3], 2, (0,1,0,1)))
        sim_env.viewer.Step()
    
    miniseg_starts, miniseg_ends = sim_util.split_trajectory_by_gripper(seg_info)    
    print colorize.colorize("mini segments:", "red"), miniseg_starts, miniseg_ends
    success = True
    feasible = True
    misgrasp = False
    full_trajs = []
    total_tps_cost = 0
    total_pose_cost = 0
    total_poses = 0
    
    for (i_miniseg, (i_start, i_end)) in enumerate(zip(miniseg_starts, miniseg_ends)):            

        ################################    
        redprint("Generating joint trajectory for part %i"%(i_miniseg))

        # figure out how we're gonna resample stuff
        lr2oldtraj = {}
        for lr in 'lr':
            manip_name = {"l":"leftarm", "r":"rightarm"}[lr]                 
            old_joint_traj = asarray(seg_info[manip_name][i_start:i_end+1])
            #print (old_joint_traj[1:] - old_joint_traj[:-1]).ptp(axis=0), i_start, i_end
            if sim_util.arm_moved(old_joint_traj):       
                lr2oldtraj[lr] = old_joint_traj   
        if len(lr2oldtraj) > 0:
            old_total_traj = np.concatenate(lr2oldtraj.values(), 1)
            JOINT_LENGTH_PER_STEP = .1
            _, timesteps_rs = sim_util.unif_resample(old_total_traj, JOINT_LENGTH_PER_STEP)
        ####

        ### Generate fullbody traj
        lr2newtraj = {}
        redprint("Optimizing JOINT ANGLE trajectory for part %i"%i_miniseg)
        for (lr,old_joint_traj) in lr2oldtraj.items():
            
            manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
            
            old_joint_traj_rs = mu.interp2d(timesteps_rs, np.arange(len(old_joint_traj)), old_joint_traj)
            
            ee_link_name = "%s_gripper_tool_frame"%lr
            new_ee_traj = lr2eetraj[lr][i_start:i_end+1]          
            new_ee_traj_rs = resampling.interp_hmats(timesteps_rs, np.arange(len(new_ee_traj)), new_ee_traj)
            new_joint_traj = planning.plan_follow_traj(sim_env.robot, manip_name,
                                                       sim_env.robot.GetLink(ee_link_name), new_ee_traj_rs,old_joint_traj_rs, beta = GlobalVars.beta)[0]
            lr2newtraj[lr] = new_joint_traj
            ################################    
        redprint("Finished JOINT ANGLE trajectory for part %i using arms '%s'"%(i_miniseg, lr2newtraj.keys()))
        dof_inds = sim_util.get_full_traj(sim_env, lr2newtraj)[1]
        
        if len(lr2newtraj) > 0:
            manip_names = []
            ee_links = []
            hmat_seglist = []
            old_trajs = []
            redprint("Optimizing TPS trajectory for part %i"%i_miniseg)
            for (lr, traj) in lr2newtraj.items():
                manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
                manip_names.append(manip_name)
                ee_link_name = "%s_gripper_tool_frame"%lr
                ee_links.append(sim_env.robot.GetLink(ee_link_name))
                new_ee_traj = hmat_dict[lr][i_start:i_end+1]
                new_ee_traj_rs = resampling.interp_hmats(timesteps_rs, np.arange(len(new_ee_traj)), new_ee_traj)
                hmat_seglist.append(new_ee_traj_rs)
                old_trajs.append(traj)
           
            wt_n = corr.sum(axis=1)
            xtarg_nd = (corr/wt_n[:,None]).dot(new_cloud[:,:3])
            tps_traj, f_new, tps_cost, pose_costs = planning.joint_fit_tps_follow_traj(sim_env.robot, '+'.join(manip_names),
                                                   ee_links, f, hmat_seglist, old_trajs, old_cloud[:,:3], xtarg_nd,
                                                   alpha=GlobalVars.alpha, beta=GlobalVars.beta, bend_coef=f._bend_coef, rot_coef=f._rot_coef, wt_n=wt_n)
            tps_full_traj = (tps_traj, dof_inds)
            n_steps = len(hmat_seglist) 
            total_tps_cost += tps_cost / float(GlobalVars.alpha)  # Normalize tps cost
            total_pose_cost += pose_costs * float(n_steps) / float(GlobalVars.beta)  # Normalize pose costs
            total_poses += len(lr2newtraj) * n_steps

            if animate:
                del handles_f[:]
                handles_f.append(sim_env.env.plot3(f_new.transform_points(old_cloud[:,:3]), 5, old_cloud[:,3:] if use_color else (0,1,0)))
                handles_f.extend(draw_grid(sim_env, old_cloud[:,:3], f_new))
                for lr in lr2newtraj.keys():
                    handles_f.append(sim_env.env.drawlinestrip(f_new.transform_hmats(hmat_dict[lr][i_start:i_end+1])[:,:3,3], 2, (0,1,0,1)))
                    handles.append(sim_env.env.drawlinestrip(sim_util.get_ee_traj(sim_env, lr, tps_traj)[:,:3,3], 2, (0,0,1,1)))
                sim_env.viewer.Step()

            redprint("Finished TPS trajectory for part %i using arms '%s'"%(i_miniseg, lr2newtraj.keys()))
        else:
            tps_full_traj = (np.zeros((0,0)), [])
        full_trajs.append(tps_full_traj)
        if not simulate:
            continue

        # For open/close gripper logic to work correctly set_gripper_maybesim needs to be called even if tps_full_traj is empty
        for lr in 'lr':
            gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%lr][i_start])
            prev_gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%lr][i_start-1]) if i_start != 0 else False
            if not sim_util.set_gripper_maybesim(sim_env, lr, gripper_open, prev_gripper_open):
                redprint("Grab %s failed" % lr)
                success = False
                misgrasp = True

        if not success: break

        if len(tps_full_traj[0]) > 0:
            if not eval_util.traj_is_safe(sim_env, tps_full_traj, COLLISION_DIST_THRESHOLD):
                redprint("Trajectory not feasible")
                feasible = False
                success = False
            else:  # Only execute feasible trajectories
                success &= sim_util.sim_full_traj_maybesim(sim_env, tps_full_traj, animate=animate, interactive=interactive)

        if not success: break

    if not simulate:
        return total_tps_cost, float(total_pose_cost) / float(total_poses)

    sim_env.sim.settle(animate=animate)
    sim_env.sim.release_rope('l')
    sim_env.sim.release_rope('r')
    sim_util.reset_arms_to_side(sim_env)
    if animate:
        sim_env.viewer.Step()
    
    return success, feasible, misgrasp, full_trajs

def simulate_demo_traj(sim_env, new_cloud, action, use_color, full_trajs, animate=False, interactive=False):
    seg_info = GlobalVars.actions[action]
    sim_util.reset_arms_to_side(sim_env)

    cloud_dim = 6 if use_color else 3
    new_cloud = new_cloud[:,:cloud_dim]
    old_cloud = get_ds_cloud(sim_env, action)[:,:cloud_dim]

    if GlobalVars.gripper_weighting:
        interest_pts = get_closing_pts(seg_info)
        raise NonImplementedError
    else:
        interest_pts = None
    f, corr = tps_registration.tps_rpm(old_cloud[:,:3], new_cloud[:,:3], vis_cost_xy = tps_registration.ab_cost(old_cloud, new_cloud) if use_color else None, user_data={'old_cloud':old_cloud, 'new_cloud':new_cloud})

    handles = []
    if animate:
        handles.append(sim_env.env.plot3(old_cloud[:,:3], 5, (1,0,0)))
        handles.append(sim_env.env.plot3(new_cloud[:,:3], 5, new_cloud[:,3:] if use_color else (0,0,1)))
        handles.append(sim_env.env.plot3(f.transform_points(old_cloud[:,:3]), 5, old_cloud[:,3:] if use_color else (0,1,0)))
        handles.extend(draw_grid(sim_env, old_cloud[:,:3], f))
        for lr in 'lr':
            link_name = "%s_gripper_tool_frame"%lr
            old_ee_traj = asarray(seg_info[link_name]["hmat"])
            handles.append(sim_env.env.drawlinestrip(old_ee_traj[:,:3,3], 2, (1,0,0,1)))
            handles.append(sim_env.env.drawlinestrip(f.transform_hmats(old_ee_traj)[:,:3,3], 2, (0,1,0,1)))
        sim_env.viewer.Step()

    miniseg_starts, miniseg_ends = sim_util.split_trajectory_by_gripper(seg_info)    
    success = True
    feasible = True
    misgrasp = False
    print colorize.colorize("mini segments:", "red"), miniseg_starts, miniseg_ends

    for (i_miniseg, (i_start, i_end)) in enumerate(zip(miniseg_starts, miniseg_ends)):      
        if i_miniseg >= len(full_trajs): break           

        full_traj = full_trajs[i_miniseg]

        if animate:
            if len(full_traj[0]) > 0: # the robot's arm might not move but the grippers might still open or close
                for lr in 'lr':
                    handles.append(sim_env.env.drawlinestrip(sim_util.get_ee_traj(sim_env, lr, full_traj)[:,:3,3], 2, (0,0,1,1)))
            sim_env.viewer.Step()

        for lr in 'lr':
            gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%lr][i_start])
            prev_gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%lr][i_start-1]) if i_start != 0 else False
            if not sim_util.set_gripper_maybesim(sim_env, lr, gripper_open, prev_gripper_open):
                redprint("Grab %s failed" % lr)
                misgrasp = True
                success = False

        if not success: break

        if len(full_traj[0]) > 0:
            if not eval_util.traj_is_safe(sim_env, full_traj, COLLISION_DIST_THRESHOLD):
                redprint("Trajectory not feasible")
                feasible = False
                success = False
            else:  # Only execute feasible trajectories
                success &= sim_util.sim_full_traj_maybesim(sim_env, full_traj, animate=animate, interactive=interactive)

        if not success: break

    sim_env.sim.settle(animate=animate)
    sim_env.sim.release_rope('l')
    sim_env.sim.release_rope('r')
    sim_util.reset_arms_to_side(sim_env)
    if animate:
        sim_env.viewer.Step()
    
    return success, feasible, misgrasp, full_trajs

def regcost_feature_fn(sim_env, state, action, use_color):
    new_cloud = state[1]
    old_cloud = get_ds_cloud(sim_env, action)
    f, corr = tps_registration.tps_rpm(old_cloud[:,:3], new_cloud[:,:3], n_iter=8, em_iter=1, vis_cost_xy = tps_registration.ab_cost(old_cloud, new_cloud) if use_color else None, user_data={'old_cloud':old_cloud, 'new_cloud':new_cloud})
    cost = registration.tps_reg_cost(f)
    return np.array([float(cost)]) # no need to normalize since bending cost is independent of number of points
   
def regcost_trajopt_feature_fn(sim_env, state, action):
    regcost = registration_cost_cheap(state[1], get_ds_cloud(sim_env, action))
    pose_cost = compute_trans_traj(sim_env, state[1], GlobalVars.actions[action], simulate=False)

    # don't rescale by alpha and beta
    # pose_cost should already be normalized by 1/n_steps and rescaled by 1/beta
    # (original trajopt cost has constant multiplier of beta/n_steps)
    print "Regcost:", float(regcost) / get_ds_cloud(sim_env, action).shape[0], "Total", float(regcost) / get_ds_cloud(sim_env, action).shape[0] + float(pose_cost)
    GlobalVars.tps_errors_top10.append(float(regcost) / get_ds_cloud(sim_env, action).shape[0])
    GlobalVars.trajopt_errors_top10.append(float(pose_cost))
    print "tps_cost, tps_pose_cost", float(regcost) / get_ds_cloud(sim_env, action).shape[0], pose_cost
    return np.array([float(regcost) / get_ds_cloud(sim_env, action).shape[0] + float(pose_cost)])  # TODO: Consider regcost + C*err

def regcost_trajopt_tps_feature_fn(sim_env, state, action):
    link_names = ["%s_gripper_tool_frame"%lr for lr in ('lr')]
    new_pts = state[1]
    demo_pts = get_ds_cloud(sim_env, action)

    # TrajOpt 
    target_trajs = warp_hmats(get_ds_cloud(sim_env, action), state[1],[(lr, GlobalVars.actions[action][ln]['hmat']) for lr, ln in zip('lr', link_names)], None)[0]
    orig_joint_trajs = traj_utils.joint_trajs(action, GlobalVars.actions)
    feasible_trajs, err = follow_trajectory_cost(sim_env, target_trajs, orig_joint_trajs, sim_env.robot)

    arm_inds = {}
    ee_links = {}
    manip_names = {"l":"leftarm", "r":"rightarm"}
    for lr in 'lr':
        arm_inds[lr] = sim_env.robot.GetManipulator(manip_names[lr]).GetArmIndices()
        ee_link_name = "%s_gripper_tool_frame"%lr
        ee_links[lr] = sim_env.robot.GetLink(ee_link_name)

    # Add trajectory positions to new_pts and demo_pts, before calling registration_cost_cheap
    orig_traj_positions = []
    feasible_traj_positions = []
    with openravepy.RobotStateSaver(sim_env.robot):
        for lr in 'lr': 
            for i_step in range(orig_joint_trajs[lr].shape[0]):
                sim_env.robot.SetDOFValues(orig_joint_trajs[lr][i_step], arm_inds[lr])
                tf = ee_links[lr].GetTransform()
                orig_traj_positions.append(tf[:3,3])
                sim_env.robot.SetDOFValues(feasible_trajs[lr][i_step], arm_inds[lr])
                tf = ee_links[lr].GetTransform()
                feasible_traj_positions.append(tf[:3,3])

    new_pts_traj = np.r_[new_pts, np.array(feasible_traj_positions)]
    demo_pts_traj = np.r_[demo_pts, np.array(orig_traj_positions)]
    return np.array([registration_cost(new_pts_traj, demo_pts_traj)[4]])

def jointopt_feature_fn(sim_env, state, action):
    # Interfaces with the jointopt code to return a cost (corresponding to the value
    # of the objective function)
    tps_cost, tps_pose_cost = compute_trans_traj_jointopt(sim_env, state[1], GlobalVars.actions[action], simulate=False)
    print "tps_cost, tps_pose_cost", tps_cost, tps_pose_cost
    return np.array([tps_cost + tps_pose_cost])

def q_value_fn(state, action, sim_env, fn):
    return np.dot(WEIGHTS, fn(sim_env, state, action)) #+ w0

def setup_log_file(args):
    if args.log:
        redprint("Writing log to file %s" % args.log)
        GlobalVars.exec_log = task_execution.ExecutionLog(args.log)
        atexit.register(GlobalVars.exec_log.close)
        GlobalVars.exec_log(0, "main.args", args)

def set_global_vars(args, sim_env):
    if args.random_seed is not None: np.random.seed(args.random_seed)

    # Note: alpha and beta should not be changed anywhere else!
    if args.subparser_name == "eval":
        GlobalVars.alpha = args.alpha
        GlobalVars.beta = args.beta

    GlobalVars.actions = h5py.File(args.actionfile, 'r')
    if args.subparser_name == "eval":
        GlobalVars.gripper_weighting = args.gripper_weighting
    
def select_feature_fn(warping_cost, args):
    if warping_cost == "regcost":
        feature_fn = lambda sim_env, state, action: regcost_feature_fn(sim_env, state, action, args.use_color)
    elif warping_cost == "regcost-trajopt":
        feature_fn = lambda sim_env, state, action: regcost_trajopt_feature_fn(sim_env, state, action, args.use_color)
    elif warping_cost == "regcost-trajopt-tps":
        feature_fn = lambda sim_env, state, action: regcost_trajopt_tps_feature_fn(sim_env, state, action, args.use_color)
    else:
        feature_fn = lambda sim_env, state, action: jointopt_feature_fn(sim_env, state, action, args.use_color)
    return feature_fn

def select_traj_fn(args):
    # compute_traj_fn is used to compute the transferred trajectory
    if args.jointopt:
        compute_traj_fn = compute_trans_traj_jointopt
    else:
        compute_traj_fn = compute_trans_traj
    return compute_traj_fn

def parse_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('actionfile', type=str, nargs='?', default='data/misc/actions.h5')
    parser.add_argument('holdoutfile', type=str, nargs='?', default='data/misc/holdout_set.h5')

    parser.add_argument("--animation", type=int, default=0, help="if greater than 1, the viewer tries to load the window and camera properties without idling at the beginning")
    parser.add_argument("--interactive", action="store_true", help="step animation and optimization if specified")
    parser.add_argument("--obstacles", type=str, nargs='*', choices=['bookshelve', 'boxes', 'cylinders'], default=[])
    parser.add_argument("--resultfile", type=str, help="no results are saved if this is not specified")

    # selects tasks to evaluate/replay
    parser.add_argument("--tasks", type=int, nargs='*', metavar="i_task")
    parser.add_argument("--taskfile", type=str)
    parser.add_argument("--i_start", type=int, default=-1, metavar="i_task")
    parser.add_argument("--i_end", type=int, default=-1, metavar="i_task")
    
    parser.add_argument("--camera_matrix_file", type=str, default='.camera_matrix.txt')
    parser.add_argument("--window_prop_file", type=str, default='.win_prop.txt')
    parser.add_argument("--fake_data_segment",type=str, default='demo1-seg00')
    parser.add_argument("--fake_data_transform", type=float, nargs=6, metavar=("tx","ty","tz","rx","ry","rz"),
        default=[0,0,0,0,0,0], help="translation=(tx,ty,tz), axis-angle rotation=(rx,ry,rz)")
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--log", type=str, default="")
    parser.add_argument("--print_mean_and_var", action="store_true")

    subparsers = parser.add_subparsers(dest='subparser_name')

    parser_eval = subparsers.add_parser('eval')
    parser_eval.add_argument('warpingcost', type=str, choices=['regcost', 'regcost-trajopt', 'regcost-trajopt-tps', 'jointopt'])
    parser_eval.add_argument("--jointopt", action="store_true")
    parser_eval.add_argument("--search_until_feasible", action="store_true")
    parser_eval.add_argument("--alpha", type=int, default=20)
    parser_eval.add_argument("--beta", type=int, default=10)
    parser_eval.add_argument("--gripper_weighting", action="store_true")
    parser_eval.add_argument("--num_steps", type=int, default=5, help="maximum number of steps to simulate each task")
    parser_eval.add_argument("--use_color", type=int, default=1)
    
    parser_replay = subparsers.add_parser('replay')
    parser_replay.add_argument("loadresultfile", type=str)
    parser_replay.add_argument("--compute_traj_steps", type=int, default=[], nargs='*', metavar='i_step', help="recompute trajectories for the i_step of all tasks")

    return parser.parse_args()

def get_unique_id(): 
    GlobalVars.unique_id += 1
    return GlobalVars.unique_id - 1

def select_best_action(sim_env, state, num_actions_to_try, feature_fn, prune_feature_fn, eval_stats, warpingcost):
    redprint("Choosing an action")
    num_top_actions = max(num_actions_to_try, TRAJOPT_MAX_ACTIONS)

    start_time = time.time()
    q_values_prune = [(q_value_fn(state, action, sim_env, prune_feature_fn), action) for action in GlobalVars.actions]
    agenda_top_actions = sorted(q_values_prune, key = lambda v: -v[0])[:num_top_actions]

    if feature_fn == prune_feature_fn:
        q_values = q_values_prune
        agenda = agenda_top_actions
    else:
        q_values = [(q_value_fn(state, a, sim_env, feature_fn), a) for (v, a) in agenda_top_actions]
        agenda = sorted(q_values, key = lambda v: -v[0])

    eval_stats.action_elapsed_time += time.time() - start_time
    q_values_root = [q for (q, a) in q_values]
    return agenda, q_values_root

def eval_on_holdout(args, sim_env):
    feature_fn = select_feature_fn(args.warpingcost, args)
    if args.warpingcost == 'regcost':
        prune_feature_fn = feature_fn # so that we can check for function equality
    else:
        prune_feature_fn = select_feature_fn('regcost', args)
    compute_traj_fn = select_traj_fn(args)
    holdoutfile = h5py.File(args.holdoutfile, 'r')
    holdout_items = eval_util.get_holdout_items(holdoutfile, args.tasks, args.taskfile, args.i_start, args.i_end)

    num_successes = 0
    num_total = 0

    for i_task, demo_id_rope_nodes in holdout_items:
        print "task %s" % i_task
        sim_util.reset_arms_to_side(sim_env)
        redprint("Replace rope")
        rope_nodes = demo_id_rope_nodes["rope_nodes"][:]
        rope_params = 'default'
        # don't call replace_rope and sim.settle() directly. use time machine interface for deterministic results!
        time_machine = sim_util.RopeSimTimeMachine(rope_nodes, sim_env)

        if args.animation:
            sim_env.viewer.Step()

        eval_util.save_task_results_init(args.resultfile, sim_env, i_task, rope_nodes, args, rope_params)

        for i_step in range(args.num_steps):
            print "task %s step %i" % (i_task, i_step)
            sim_util.reset_arms_to_side(sim_env)
            
            redprint("Observe point cloud")
            new_cloud, endpoint_inds = sim_env.sim.raycast_cloud(endpoints=3)
            if new_cloud.shape[0] == 0: # rope is not visible (probably because it fall off the table)
                break
            if args.use_color:
                new_cloud = color_cloud(new_cloud, endpoint_inds)
            new_cloud_ds = clouds.downsample(new_cloud, DS_SIZE)
            state = ("eval_%i"%get_unique_id(), new_cloud_ds)
    
            num_actions_to_try = MAX_ACTIONS_TO_TRY if args.search_until_feasible else 1
            eval_stats = eval_util.EvalStats()

            agenda, q_values_root = select_best_action(sim_env, state, num_actions_to_try, feature_fn, prune_feature_fn, eval_stats, args.warpingcost)

            time_machine.set_checkpoint('prechoice_%i'%i_step, sim_env)
            for i_choice in range(num_actions_to_try):
                time_machine.restore_from_checkpoint('prechoice_%i'%i_step, sim_env, sim_util.get_rope_params(rope_params))
                best_root_action = agenda[i_choice][1]
                start_time = time.time()
                eval_stats.success, eval_stats.feasible, eval_stats.misgrasp, full_trajs = compute_traj_fn(sim_env, new_cloud_ds, best_root_action, args.use_color, animate=args.animation, interactive=args.interactive)
                eval_stats.exec_elapsed_time += time.time() - start_time

                if eval_stats.feasible:  # try next action if TrajOpt cannot find feasible action
                     eval_stats.found_feasible_action = True
                     break
                else:
                     redprint('TRYING NEXT ACTION')

            if not eval_stats.feasible:  # If not feasible, restore_from_checkpoint
                time_machine.restore_from_checkpoint('prechoice_%i'%i_step, sim_env, sim_util.get_rope_params(rope_params))
            print "BEST ACTION:", best_root_action
                
            demo_cloud = GlobalVars.actions[best_root_action]['cloud_xyz'][()]
            demo_cloud_ds = get_ds_cloud(sim_env, best_root_action)
            eval_util.save_task_results_step(args.resultfile, sim_env, i_task, i_step, eval_stats, best_root_action, full_trajs, q_values_root, demo_cloud, demo_cloud_ds, new_cloud, new_cloud_ds)
            
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
    holdoutfile = h5py.File(args.holdoutfile, 'r')
    loadresultfile = h5py.File(args.loadresultfile, 'r')
    loadresult_items = eval_util.get_holdout_items(loadresultfile, args.tasks, args.taskfile, args.i_start, args.i_end)

    num_successes = 0
    num_total = 0
    
    for i_task, _ in loadresult_items:
        print "task %s" % i_task
        sim_util.reset_arms_to_side(sim_env)
        redprint("Replace rope")
        rope_nodes, rope_params, loaded_args, _, _ = eval_util.load_task_results_init(args.loadresultfile, i_task)
        # don't call replace_rope and sim.settle() directly. use time machine interface for deterministic results!
        time_machine = sim_util.RopeSimTimeMachine(rope_nodes, sim_env)

        if args.animation:
            sim_env.viewer.Step()

        eval_util.save_task_results_init(args.resultfile, sim_env, i_task, rope_nodes, args, rope_params)
        
        for i_step in range(len(loadresultfile[i_task]) - (1 if 'init' in loadresultfile[i_task] else 0)):
            print "task %s step %i" % (i_task, i_step)
            sim_util.reset_arms_to_side(sim_env)

            redprint("Observe point cloud")
            new_cloud, endpoint_inds = sim_env.sim.raycast_cloud(endpoints=3)
            if new_cloud.shape[0] == 0: # rope is not visible (probably because it fall off the table)
                break
            if loaded_args.use_color:
                new_cloud = color_cloud(new_cloud, endpoint_inds)
            new_cloud_ds = clouds.downsample(new_cloud, DS_SIZE)
            state = ("eval_%i"%get_unique_id(), new_cloud_ds)
    
            eval_stats = eval_util.EvalStats()

            best_action, full_trajs, q_values, trans, rots = eval_util.load_task_results_step(args.loadresultfile, sim_env, i_task, i_step)
            
            time_machine.set_checkpoint('preexec_%i'%i_step, sim_env)
            time_machine.restore_from_checkpoint('preexec_%i'%i_step, sim_env, sim_util.get_rope_params(rope_params))
            start_time = time.time()
            if i_step in args.compute_traj_steps:
                compute_traj_fn = select_traj_fn(loaded_args)
                eval_stats.success, eval_stats.feasible, eval_stats.misgrasp, full_trajs = compute_traj_fn(sim_env, new_cloud_ds, best_action, loaded_args.use_color, animate=args.animation, interactive=args.interactive)
            else:
                eval_stats.success, eval_stats.feasible, eval_stats.misgrasp, full_trajs = simulate_demo_traj(sim_env, new_cloud_ds, best_action, loaded_args.use_color, full_trajs, animate=args.animation, interactive=args.interactive)
            eval_stats.exec_elapsed_time += time.time() - start_time

            if eval_stats.feasible:
                 eval_stats.found_feasible_action = True

            if not eval_stats.feasible:  # If not feasible, restore_from_checkpoint
                time_machine.restore_from_checkpoint('preexec_%i'%i_step, sim_env, sim_util.get_rope_params(rope_params))
            print "BEST ACTION:", best_action

            replay_trans, replay_rots = sim_util.get_rope_transforms(sim_env)
            if np.linalg.norm(trans - replay_trans) > 0 or np.linalg.norm(rots - replay_rots) > 0:
                yellowprint("The rope transforms of the replay rope doesn't match the ones in the original result file by %f and %f" % (np.linalg.norm(trans - replay_trans), np.linalg.norm(rots - replay_rots)))
            else:
                yellowprint("Reproducible results OK")
            
            demo_cloud = GlobalVars.actions[best_action]['cloud_xyz'][()]
            demo_cloud_ds = get_ds_cloud(sim_env, best_action)
            eval_util.save_task_results_step(args.resultfile, sim_env, i_task, i_step, eval_stats, best_action, full_trajs, q_values, demo_cloud, demo_cloud_ds, new_cloud, new_cloud_ds)
            
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

def load_simulation(args, sim_env):
    sim_env.env = openravepy.Environment()
    sim_env.env.StopSimulation()
    sim_env.env.Load("robots/pr2-beta-static.zae")
    sim_env.robot = sim_env.env.GetRobots()[0]

    actions = h5py.File(args.actionfile, 'r')
    
    init_rope_xyz, _ = sim_util.load_fake_data_segment(sim_env, actions, args.fake_data_segment, args.fake_data_transform) # this also sets the torso (torso_lift_joint) to the height in the data
    table_height = init_rope_xyz[:,2].mean() - .02
    table_xml = sim_util.make_table_xml(translation=[1, 0, table_height], extents=[.85, .55, .01])
    sim_env.env.LoadData(table_xml)
    obstacle_bodies = []
    if 'bookshelve' in args.obstacles:
        sim_env.env.Load("data/bookshelves.env.xml")
        obstacle_bodies.extend(sim_env.env.GetBodies()[-1:])
    if 'boxes' in args.obstacles:
        sim_env.env.LoadData(sim_util.make_box_xml("box0", [.7,.43,table_height+(.01+.12)], [.12,.12,.12]))
        sim_env.env.LoadData(sim_util.make_box_xml("box1", [.74,.47,table_height+(.01+.12*2+.08)], [.08,.08,.08]))
        obstacle_bodies.extend(sim_env.env.GetBodies()[-2:])
    if 'cylinders' in args.obstacles:
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

def main():
    args = parse_input_args()
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
