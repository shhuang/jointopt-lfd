#!/usr/bin/env python

import pprint
import argparse
import planning, tps_registration

from rapprentice import registration, colorize, berkeley_pr2, \
     animate_traj, ros2rave, plotting_openrave, task_execution, \
     tps, func_utils, resampling, ropesim, rope_initialization, clouds
from rapprentice import math_utils as mu
from rapprentice.yes_or_no import yes_or_no
import pdb, time

try:
    from rapprentice import pr2_trajectories, PR2
    import rospy
except ImportError:
    print "Couldn't import ros stuff"

import cloudprocpy, trajoptpy, openravepy
import util
from rope_qlearn import *
from knot_classifier import isKnot as is_knot
import os, numpy as np, h5py
from numpy import asarray
import atexit
import importlib
from itertools import combinations
import IPython as ipy
import random

COLLISION_DIST_THRESHOLD = 0.01

def traj_collisions(traj, n=100):
    """
    Returns the set of collisions. 
    manip = Manipulator or list of indices
    """
    traj_up = mu.interp2d(np.linspace(0,1,n), np.linspace(0,1,len(traj)), traj)
    _ss = openravepy.RobotStateSaver(Globals.robot)

    cc = trajoptpy.GetCollisionChecker(Globals.env)
    links_to_exclude = Globals.robot.GetLinks()

    for link in links_to_exclude:
        for rope_link in Globals.env.GetKinBody('rope').GetLinks():
            cc.ExcludeCollisionPair(link, rope_link)
        cc.ExcludeCollisionPair(link, Globals.env.GetKinBody('table').GetLinks()[0])
    
    col_times = []
    for (i,row) in enumerate(traj_up):
        Globals.robot.SetActiveDOFValues(row)
        col_now = cc.BodyVsAll(Globals.robot)
        #with util.suppress_stdout():
        #    col_now2 = cc.PlotCollisionGeometry()
        col_now = [cn for cn in col_now if cn.GetDistance() < COLLISION_DIST_THRESHOLD]
        if col_now:
            print [cn.GetDistance() for cn in col_now]
            col_times.append(i)
            print "trajopt.CollisionChecker: ", len(col_now)
        #print col_now2
        
    return col_times

def traj_is_safe(traj, n=100):
    return traj_collisions(traj, n) == []

def redprint(msg):
    print colorize.colorize(msg, "red", bold=True)
    
def split_trajectory_by_gripper(seg_info):
    rgrip = asarray(seg_info["r_gripper_joint"])
    lgrip = asarray(seg_info["l_gripper_joint"])

    thresh = .04 # open/close threshold

    n_steps = len(lgrip)


    # indices BEFORE transition occurs
    l_openings = np.flatnonzero((lgrip[1:] >= thresh) & (lgrip[:-1] < thresh))
    r_openings = np.flatnonzero((rgrip[1:] >= thresh) & (rgrip[:-1] < thresh))
    l_closings = np.flatnonzero((lgrip[1:] < thresh) & (lgrip[:-1] >= thresh))
    r_closings = np.flatnonzero((rgrip[1:] < thresh) & (rgrip[:-1] >= thresh))

    before_transitions = np.r_[l_openings, r_openings, l_closings, r_closings]
    after_transitions = before_transitions+1
    seg_starts = np.unique(np.r_[0, after_transitions])
    seg_ends = np.unique(np.r_[before_transitions, n_steps-1])

    return seg_starts, seg_ends

def binarize_gripper(angle):
    thresh = .04
    return angle > thresh
    
def set_gripper_maybesim(lr, is_open, prev_is_open):
    mult = 5
    open_angle = .08 * mult
    closed_angle = .02 * mult

    target_val = open_angle if is_open else closed_angle

    # release constraints if necessary
    if is_open and not prev_is_open:
        Globals.sim.release_rope(lr)
        print "DONE RELEASING"

    # execute gripper open/close trajectory
    joint_ind = Globals.robot.GetJoint("%s_gripper_l_finger_joint"%lr).GetDOFIndex()
    start_val = Globals.robot.GetDOFValues([joint_ind])[0]
    joint_traj = np.linspace(start_val, target_val, np.ceil(abs(target_val - start_val) / .02))
    for val in joint_traj:
        Globals.robot.SetDOFValues([val], [joint_ind])
        Globals.sim.step()
#         if args.animation:
#                Globals.viewer.Step()
#             if args.interactive: Globals.viewer.Idle()
    # add constraints if necessary
    if Globals.viewer:
        Globals.viewer.Step()
    if not is_open and prev_is_open:
        if not Globals.sim.grab_rope(lr):
            return False

    return True

def unwrap_arm_traj_in_place(traj):
    assert traj.shape[1] == 7
    for i in [2,4,6]:
        traj[:,i] = np.unwrap(traj[:,i])
    return traj

def unwrap_in_place(t):
    # TODO: do something smarter than just checking shape[1]
    if t.shape[1] == 7:
        unwrap_arm_traj_in_place(t)
    elif t.shape[1] == 14:
        unwrap_arm_traj_in_place(t[:,:7])
        unwrap_arm_traj_in_place(t[:,7:])
    else:
        raise NotImplementedError

def sim_traj_maybesim(bodypart2traj, animate=False, interactive=False):
    dof_inds = []
    trajs = []
    for (part_name, traj) in bodypart2traj.items():
        manip_name = {"larm":"leftarm","rarm":"rightarm"}[part_name]
        dof_inds.extend(Globals.robot.GetManipulator(manip_name).GetArmIndices())            
        trajs.append(traj)
    full_traj = np.concatenate(trajs, axis=1)
    Globals.robot.SetActiveDOFs(dof_inds)
    sim_full_traj_maybesim(full_traj, dof_inds, animate=animate, interactive=interactive)

def sim_full_traj_maybesim(full_traj, dof_inds, animate=False, interactive=False):
    def sim_callback(i):
        Globals.sim.step()

    animate_speed = 10 if animate else 0

    # make the trajectory slow enough for the simulation
    full_traj = ropesim.retime_traj(Globals.robot, dof_inds, full_traj)

    # in simulation mode, we must make sure to gradually move to the new starting position
    curr_vals = Globals.robot.GetActiveDOFValues()
    transition_traj = np.r_[[curr_vals], [full_traj[0]]]
    unwrap_in_place(transition_traj)
    transition_traj = ropesim.retime_traj(Globals.robot, dof_inds, transition_traj, max_cart_vel=.05)
    animate_traj.animate_traj(transition_traj, Globals.robot, restore=False, pause=interactive,
        callback=sim_callback, step_viewer=animate_speed)
    full_traj[0] = transition_traj[-1]
    unwrap_in_place(full_traj)

    animate_traj.animate_traj(full_traj, Globals.robot, restore=False, pause=interactive,
        callback=sim_callback, step_viewer=animate_speed)
    if Globals.viewer:
        Globals.viewer.Step()
    return True

def load_random_start_segment(demofile):
    start_keys = [k for k in demofile.keys() if k.startswith('demo') and k.endswith('00')]
    seg_name = random.choice(start_keys)
    return demofile[seg_name]['cloud_xyz']

def sample_rope_state(demofile, human_check=True, perturb_points=5, min_rad=0, max_rad=.15):
    success = False
    while not success:
        # TODO: pick a random rope initialization
        new_xyz= load_random_start_segment(demofile)
        perturb_radius = random.uniform(min_rad, max_rad)
        rope_nodes = rope_initialization.find_path_through_point_cloud( new_xyz,
                                                                        perturb_peak_dist=perturb_radius,
                                                                        num_perturb_points=perturb_points)
        replace_rope(rope_nodes)
        Globals.sim.settle()
        Globals.viewer.Step()
        if human_check:
            resp = raw_input("Use this simulation?[Y/n]")
            success = resp not in ('N', 'n')
        else:
            success = True

DS_SIZE = .025

def simulate_demo(new_xyz, seg_info, animate=False):
    Globals.robot.SetDOFValues(PR2_L_POSTURES["side"], Globals.robot.GetManipulator("leftarm").GetArmIndices())
    Globals.robot.SetDOFValues(mirror_arm_joints(PR2_L_POSTURES["side"]), Globals.robot.GetManipulator("rightarm").GetArmIndices())
    
    redprint("Generating end-effector trajectory")    
    
    handles = []
    old_xyz = np.squeeze(seg_info["cloud_xyz"])
    handles.append(Globals.env.plot3(old_xyz,5, (1,0,0)))
    handles.append(Globals.env.plot3(new_xyz,5, (0,0,1)))
    
    old_xyz = clouds.downsample(old_xyz, DS_SIZE)
    new_xyz = clouds.downsample(new_xyz, DS_SIZE)
    
    link_names = ["%s_gripper_tool_frame"%lr for lr in ('lr')]
    hmat_list = [(lr, seg_info[ln]['hmat']) for lr, ln in zip('lr', link_names)]
    if args.gripper_weighting:
        interest_pts = get_closing_pts(seg_info)
    else:
        interest_pts = None
    lr2eetraj = warp_hmats(old_xyz, new_xyz, hmat_list, interest_pts)[0]

    miniseg_starts, miniseg_ends = split_trajectory_by_gripper(seg_info)    
    success = True
    print colorize.colorize("mini segments:", "red"), miniseg_starts, miniseg_ends
    bodypart2trajs = []

    for (i_miniseg, (i_start, i_end)) in enumerate(zip(miniseg_starts, miniseg_ends)):            

        ################################    
        redprint("Generating joint trajectory for part %i"%(i_miniseg))

        # figure out how we're gonna resample stuff
        lr2oldtraj = {}
        for lr in 'lr':
            manip_name = {"l":"leftarm", "r":"rightarm"}[lr]                 
            old_joint_traj = asarray(seg_info[manip_name][i_start:i_end+1])
            #print (old_joint_traj[1:] - old_joint_traj[:-1]).ptp(axis=0), i_start, i_end
            if arm_moved(old_joint_traj):       
                lr2oldtraj[lr] = old_joint_traj   
        if len(lr2oldtraj) > 0:
            old_total_traj = np.concatenate(lr2oldtraj.values(), 1)
            JOINT_LENGTH_PER_STEP = .1
            _, timesteps_rs = unif_resample(old_total_traj, JOINT_LENGTH_PER_STEP)
        ####

        ### Generate fullbody traj
        bodypart2traj = {}

        for (lr,old_joint_traj) in lr2oldtraj.items():
            
            manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
            
            old_joint_traj_rs = mu.interp2d(timesteps_rs, np.arange(len(old_joint_traj)), old_joint_traj)
            
            ee_link_name = "%s_gripper_tool_frame"%lr
            new_ee_traj = lr2eetraj[lr][i_start:i_end+1]          
            new_ee_traj_rs = resampling.interp_hmats(timesteps_rs, np.arange(len(new_ee_traj)), new_ee_traj)
            print "planning trajectory following"
            with util.suppress_stdout():
                new_joint_traj = planning.plan_follow_traj(Globals.robot, manip_name,
                                                           Globals.robot.GetLink(ee_link_name), new_ee_traj_rs,old_joint_traj_rs)[0]

            part_name = {"l":"larm", "r":"rarm"}[lr]
            bodypart2traj[part_name] = new_joint_traj
            ################################    
            redprint("Executing joint trajectory for part %i using arms '%s'"%(i_miniseg, bodypart2traj.keys()))
        bodypart2trajs.append(bodypart2traj)
        
        for lr in 'lr':
            gripper_open = binarize_gripper(seg_info["%s_gripper_joint"%lr][i_start])
            prev_gripper_open = binarize_gripper(seg_info["%s_gripper_joint"%lr][i_start-1]) if i_start != 0 else False
            if not set_gripper_maybesim(lr, gripper_open, prev_gripper_open):
                redprint("Grab %s failed" % lr)
                success = False

        if not success: break

        if len(bodypart2traj) > 0:
            dof_inds = []
            trajs = []
            for (part_name, traj) in bodypart2traj.items():
                manip_name = {"larm":"leftarm","rarm":"rightarm"}[part_name]
                dof_inds.extend(Globals.robot.GetManipulator(manip_name).GetArmIndices())            
                trajs.append(traj)
            full_traj = np.concatenate(trajs, axis=1)
            Globals.robot.SetActiveDOFs(dof_inds)
            if not traj_is_safe(full_traj):
                redprint("Trajectory not feasible")
                success = False
                break

            success &= sim_traj_maybesim(bodypart2traj, animate=animate)

        if not success: break

    Globals.sim.settle(animate=animate)
    Globals.robot.SetDOFValues(PR2_L_POSTURES["side"], Globals.robot.GetManipulator("leftarm").GetArmIndices())
    Globals.robot.SetDOFValues(mirror_arm_joints(PR2_L_POSTURES["side"]), Globals.robot.GetManipulator("rightarm").GetArmIndices())
    if animate:
        Globals.viewer.Step()
    Globals.sim.release_rope('l')
    Globals.sim.release_rope('r')
    
    return success, bodypart2trajs

def simulate_demo_jointopt(new_xyz, seg_info, animate=False):
    Globals.robot.SetDOFValues(PR2_L_POSTURES["side"], Globals.robot.GetManipulator("leftarm").GetArmIndices())
    Globals.robot.SetDOFValues(mirror_arm_joints(PR2_L_POSTURES["side"]), Globals.robot.GetManipulator("rightarm").GetArmIndices())
    
    redprint("Generating end-effector trajectory")    
    
    handles = []
    old_xyz = np.squeeze(seg_info["cloud_xyz"])
#     handles.append(Globals.env.plot3(old_xyz,5, (1,0,0)))
#     handles.append(Globals.env.plot3(new_xyz,5, (0,0,1)))
    
    old_xyz = clouds.downsample(old_xyz, DS_SIZE)
    new_xyz = clouds.downsample(new_xyz, DS_SIZE)

    link_names = ["%s_gripper_tool_frame"%lr for lr in ('lr')]
    hmat_list = [(lr, seg_info[ln]['hmat']) for lr, ln in zip('lr', link_names)]
    hmat_dict = dict(hmat_list)
    if args.gripper_weighting:
        interest_pts = get_closing_pts(seg_info)
        raise NonImplementedError
    else:
        interest_pts = None
#     lr2eetraj = warp_hmats(old_xyz, new_xyz, hmat_list, interest_pts)[0]
    
    scaled_xyz_src, src_params = registration.unit_boxify(old_xyz)
    scaled_xyz_targ, targ_params = registration.unit_boxify(new_xyz)
    f,g, xtarg_nd, bend_coef, wt_n, rot_coef = tps_registration.tps_rpm_bij(scaled_xyz_src, scaled_xyz_targ, plot_cb=None,
                                   plotting=0, rot_reg=np.r_[1e-4, 1e-4, 1e-1], 
                                   n_iter=50, reg_init=10, reg_final=.1, outlierfrac=1e-2,
                                   x_weights=None)
    f = registration.unscale_tps(f, src_params, targ_params)
    unscaled_xtarg_nd = tps_registration.unscale_tps_points(xtarg_nd, targ_params[0], targ_params[1]) # should be close to new_xyz but with the same number of points as old_xyz

    lr2eetraj = {}
    for k, hmats in hmat_list:
        lr2eetraj[k] = f.transform_hmats(hmats)

    miniseg_starts, miniseg_ends = split_trajectory_by_gripper(seg_info)    
    print colorize.colorize("mini segments:", "red"), miniseg_starts, miniseg_ends
    success = True
    bodypart2trajs = []
    tpsfulltrajs = []
    
    for (i_miniseg, (i_start, i_end)) in enumerate(zip(miniseg_starts, miniseg_ends)):            

        ################################    
        redprint("Generating joint trajectory for part %i"%(i_miniseg))

        # figure out how we're gonna resample stuff
        lr2oldtraj = {}
        for lr in 'lr':
            manip_name = {"l":"leftarm", "r":"rightarm"}[lr]                 
            old_joint_traj = asarray(seg_info[manip_name][i_start:i_end+1])
            #print (old_joint_traj[1:] - old_joint_traj[:-1]).ptp(axis=0), i_start, i_end
            if arm_moved(old_joint_traj):       
                lr2oldtraj[lr] = old_joint_traj   
        if len(lr2oldtraj) > 0:
            old_total_traj = np.concatenate(lr2oldtraj.values(), 1)
            JOINT_LENGTH_PER_STEP = .1
            _, timesteps_rs = unif_resample(old_total_traj, JOINT_LENGTH_PER_STEP)
        ####

        ### Generate fullbody traj
        bodypart2traj = {}
        redprint("Optimizing JOINT trajectory for part %i"%i_miniseg)
        for (lr,old_joint_traj) in lr2oldtraj.items():
            
            manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
            
            old_joint_traj_rs = mu.interp2d(timesteps_rs, np.arange(len(old_joint_traj)), old_joint_traj)
            
            ee_link_name = "%s_gripper_tool_frame"%lr
            new_ee_traj = lr2eetraj[lr][i_start:i_end+1]          
            new_ee_traj_rs = resampling.interp_hmats(timesteps_rs, np.arange(len(new_ee_traj)), new_ee_traj)
            with util.suppress_stdout():
                new_joint_traj = planning.plan_follow_traj(Globals.robot, manip_name,
                                                           Globals.robot.GetLink(ee_link_name), new_ee_traj_rs,old_joint_traj_rs)[0]
            part_name = {"l":"larm", "r":"rarm"}[lr]
            bodypart2traj[part_name] = new_joint_traj
            ################################    
        redprint("Finished JOINT trajectory for part %i using arms '%s'"%(i_miniseg, bodypart2traj.keys()))
        bodypart2trajs.append(bodypart2traj)
        
        for lr in 'lr':
            gripper_open = binarize_gripper(seg_info["%s_gripper_joint"%lr][i_start])
            prev_gripper_open = binarize_gripper(seg_info["%s_gripper_joint"%lr][i_start-1]) if i_start != 0 else False
            if not set_gripper_maybesim(lr, gripper_open, prev_gripper_open):
                redprint("Grab %s failed" % lr)
                success = False
                
        if not success: break
        
        if len(bodypart2traj) > 0:
            manip_names = []
            ee_links = []
            hmat_seglist = []
            old_trajs = []
            redprint("Optimizing TPS trajectory for part %i"%i_miniseg)
            for lr in [key[0] for key in sorted(bodypart2traj.keys())]:
                part_name = {"l":"larm", "r":"rarm"}[lr]
                manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
                manip_names.append(manip_name)
                ee_link_name = "%s_gripper_tool_frame"%lr
                ee_links.append(Globals.robot.GetLink(ee_link_name))
                new_ee_traj = hmat_dict[lr][i_start:i_end+1]
                new_ee_traj_rs = resampling.interp_hmats(timesteps_rs, np.arange(len(new_ee_traj)), new_ee_traj)
                hmat_seglist.append(new_ee_traj_rs)
                old_trajs.append(bodypart2traj[part_name])
           
            tpsfulltraj, _, _ = planning.joint_fit_tps_follow_traj(Globals.robot, '+'.join(manip_names),
                                                   ee_links, f, hmat_seglist, old_trajs, old_xyz, unscaled_xtarg_nd,
                                                   alpha=1.0, bend_coef=bend_coef, rot_coef = rot_coef, wt_n=wt_n)
            redprint("Finished TPS trajectory for part %i using arms '%s'"%(i_miniseg, bodypart2traj.keys()))
            tpsfulltrajs.append(tpsfulltraj)
    
            dof_inds = []
            for manip_name in manip_names:
                dof_inds.extend(Globals.robot.GetManipulator(manip_name).GetArmIndices())            
            Globals.robot.SetActiveDOFs(dof_inds)
            if not traj_is_safe(tpsfulltraj):
                redprint("Trajectory not feasible")
                success = False
                break
            success &= sim_full_traj_maybesim(tpsfulltraj, dof_inds, animate=animate)

        if not success: break

    Globals.sim.settle(animate=animate)
    Globals.robot.SetDOFValues(PR2_L_POSTURES["side"], Globals.robot.GetManipulator("leftarm").GetArmIndices())
    Globals.robot.SetDOFValues(mirror_arm_joints(PR2_L_POSTURES["side"]), Globals.robot.GetManipulator("rightarm").GetArmIndices())
    if animate:
        Globals.viewer.Step()
    Globals.sim.release_rope('l')
    Globals.sim.release_rope('r')
    
    return success, tpsfulltrajs

def simulate_demo_traj(new_xyz, seg_info, bodypart2trajs, animate=False):
    Globals.robot.SetDOFValues(PR2_L_POSTURES["side"], Globals.robot.GetManipulator("leftarm").GetArmIndices())
    Globals.robot.SetDOFValues(mirror_arm_joints(PR2_L_POSTURES["side"]), Globals.robot.GetManipulator("rightarm").GetArmIndices())
    
    handles = []
    old_xyz = np.squeeze(seg_info["cloud_xyz"])
    handles.append(Globals.env.plot3(old_xyz,5, (1,0,0)))
    handles.append(Globals.env.plot3(new_xyz,5, (0,0,1)))
    
    miniseg_starts, miniseg_ends = split_trajectory_by_gripper(seg_info)    
    success = True
    print colorize.colorize("mini segments:", "red"), miniseg_starts, miniseg_ends
    for (i_miniseg, (i_start, i_end)) in enumerate(zip(miniseg_starts, miniseg_ends)):            
        if i_miniseg >= len(bodypart2trajs): break

        bodypart2traj = bodypart2trajs[i_miniseg]

        for lr in 'lr':
            gripper_open = binarize_gripper(seg_info["%s_gripper_joint"%lr][i_start])
            prev_gripper_open = binarize_gripper(seg_info["%s_gripper_joint"%lr][i_start-1]) if i_start != 0 else False
            if not set_gripper_maybesim(lr, gripper_open, prev_gripper_open):
                redprint("Grab %s failed" % lr)
                success = False

        if not success: break

        if len(bodypart2traj) > 0:
            success &= sim_traj_maybesim(bodypart2traj, animate=animate)

        if not success: break

    Globals.sim.settle(animate=animate)
    Globals.robot.SetDOFValues(PR2_L_POSTURES["side"], Globals.robot.GetManipulator("leftarm").GetArmIndices())
    Globals.robot.SetDOFValues(mirror_arm_joints(PR2_L_POSTURES["side"]), Globals.robot.GetManipulator("rightarm").GetArmIndices())

    Globals.sim.release_rope('l')
    Globals.sim.release_rope('r')
    
    return success

def replace_rope(new_rope):
    import bulletsimpy
    old_rope_nodes = Globals.sim.rope.GetControlPoints()
    if Globals.viewer:
        Globals.viewer.RemoveKinBody(Globals.env.GetKinBody('rope'))
    Globals.env.Remove(Globals.env.GetKinBody('rope'))
    Globals.sim.bt_env.Remove(Globals.sim.bt_env.GetObjectByName('rope'))
    Globals.sim.rope = bulletsimpy.CapsuleRope(Globals.sim.bt_env, 'rope', new_rope,
                                               Globals.sim.rope_params)
    return old_rope_nodes

def get_rope_transforms():
    return (Globals.sim.rope.GetTranslations(), Globals.sim.rope.GetRotations())    

def set_rope_transforms(tfs):
    Globals.sim.rope.SetTranslations(tfs[0])
    Globals.sim.rope.SetRotations(tfs[1])

def arm_moved(joint_traj):    
    if len(joint_traj) < 2: return False
    return ((joint_traj[1:] - joint_traj[:-1]).ptp(axis=0) > .01).any()
        
def tpsrpm_plot_cb(x_nd, y_md, targ_Nd, corr_nm, wt_n, f):
    ypred_nd = f.transform_points(x_nd)
    handles = []
    handles.append(Globals.env.plot3(ypred_nd, 3, (0,1,0,1)))
    handles.extend(plotting_openrave.draw_grid(Globals.env, f.transform_points, x_nd.min(axis=0), x_nd.max(axis=0), xres = .1, yres = .1, zres = .04))
    if Globals.viewer:
        Globals.viewer.Step()

def load_fake_data_segment(demofile, fake_data_segment, fake_data_transform, set_robot_state=True):
    fake_seg = demofile[fake_data_segment]
    new_xyz = np.squeeze(fake_seg["cloud_xyz"])
    hmat = openravepy.matrixFromAxisAngle(fake_data_transform[3:6])
    hmat[:3,3] = fake_data_transform[0:3]
    new_xyz = new_xyz.dot(hmat[:3,:3].T) + hmat[:3,3][None,:]
    r2r = ros2rave.RosToRave(Globals.robot, asarray(fake_seg["joint_states"]["name"]))
    if set_robot_state:
        r2r.set_values(Globals.robot, asarray(fake_seg["joint_states"]["position"][0]))
    return new_xyz, r2r

def get_ds_cloud(action):
    return clouds.downsample(Globals.actionfile[action]['cloud_xyz'], DS_SIZE)

def unif_resample(traj, max_diff, wt = None):        
    """
    Resample a trajectory so steps have same length in joint space    
    """
    import scipy.interpolate as si
    tol = .005
    if wt is not None: 
        wt = np.atleast_2d(wt)
        traj = traj*wt
        
        
    dl = mu.norms(traj[1:] - traj[:-1],1)
    l = np.cumsum(np.r_[0,dl])
    goodinds = np.r_[True, dl > 1e-8]
    deg = min(3, sum(goodinds) - 1)
    if deg < 1: return traj, np.arange(len(traj))
    
    nsteps = max(int(np.ceil(float(l[-1])/max_diff)), 2)
    newl = np.linspace(0,l[-1],nsteps)

    ncols = traj.shape[1]
    colstep = 10
    traj_rs = np.empty((nsteps,ncols)) 
    for istart in xrange(0, traj.shape[1], colstep):
        (tck,_) = si.splprep(traj[goodinds, istart:istart+colstep].T,k=deg,s = tol**2*len(traj),u=l[goodinds])
        traj_rs[:,istart:istart+colstep] = np.array(si.splev(newl,tck)).T
    if wt is not None: traj_rs = traj_rs/wt

    newt = np.interp(newl, l, np.arange(len(traj)))

    return traj_rs, newt

def make_table_xml(translation, extents):
    xml = """
<Environment>
  <KinBody name="table">
    <Body type="static" name="table_link">
      <Geom type="box">
        <Translation>%f %f %f</Translation>
        <extents>%f %f %f</extents>
        <diffuseColor>.96 .87 .70</diffuseColor>
      </Geom>
    </Body>
  </KinBody>
</Environment>
""" % (translation[0], translation[1], translation[2], extents[0], extents[1], extents[2])
    return xml

PR2_L_POSTURES = dict(
    untucked = [0.4,  1.0,   0.0,  -2.05,  0.0,  -0.1,  0.0],
    tucked = [0.06, 1.25, 1.79, -1.68, -1.73, -0.10, -0.09],
    up = [ 0.33, -0.35,  2.59, -0.15,  0.59, -1.41, -0.27],
    side = [  1.832,  -0.332,   1.011,  -1.437,   1.1  ,  -2.106,  3.074]
)
def mirror_arm_joints(x):
    "mirror image of joints (r->l or l->r)"
    return np.r_[-x[0],x[1],-x[2],x[3],-x[4],x[5],-x[6]]

def reset_arms_to_side():
    Globals.robot.SetDOFValues(PR2_L_POSTURES["side"],
                               Globals.robot.GetManipulator("leftarm").GetArmIndices())
    #actionfile = None
    Globals.robot.SetDOFValues(mirror_arm_joints(PR2_L_POSTURES["side"]),
                               Globals.robot.GetManipulator("rightarm").GetArmIndices())

def regcost_feature_fn(state, action):
    regcost = registration_cost_cheap(state[1], get_ds_cloud(action))
    return np.array([regcost])
   
def regcost_trajopt_feature_fn(state, action):
    link_names = ["%s_gripper_tool_frame"%lr for lr in ('lr')]
    regcost = registration_cost_cheap(state[1], get_ds_cloud(action))
    target_trajs = warp_hmats(get_ds_cloud(action), state[1],[(lr, Globals.actionfile[action][ln]['hmat']) for lr, ln in zip('lr', link_names)], None)[0]
    orig_joint_trajs = traj_utils.joint_trajs(action, Globals.actionfile)
    err = traj_utils.follow_trajectory_cost(target_trajs, orig_joint_trajs, Globals.robot)
    return np.array([float(regcost) / get_ds_cloud(action).shape[0] + \
                     float(err) / len(orig_joint_trajs.values()[0])])  # TODO: Consider regcost + C*err

def regcost_jointop_feature_fn(state, action):
    # TODO: Interface this with the jointopt code
    print "NOT IMPLEMENTED YET"

###################

class Globals:
    robot = None
    env = None
    pr2 = None
    sim = None
    log = None
    viewer = None
    resample_rope = None
    actionfile = None

if __name__ == "__main__":
    """
    example command:
    ./do_task_eval.py data/weights/multi_quad_weights_10000.h5 --quad_features --animation=1
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('actionfile', nargs='?', default='data/misc/actions.h5')
    parser.add_argument('holdoutfile', nargs='?', default='data/misc/holdout_set.h5')
    parser.add_argument('warpingcost', choices=['regcost', 'regcost-trajopt', 'jointopt'])
    parser.add_argument("--resultfile", type=str) # don't save results if this is not specified
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument("--animation", type=int, default=0)
    parser.add_argument("--i_start", type=int, default=-1)
    parser.add_argument("--i_end", type=int, default=-1)
    parser.add_argument("--lookahead_width", type=int, default=1)
    parser.add_argument("--lookahead_depth", type=int, default=0)
    parser.add_argument("--gripper_weighting", action="store_true")

    parser.add_argument("--elbow_obstacle", action="store_true")
    parser.add_argument("--jointopt", action="store_true")
    
    parser.add_argument("--tasks", nargs='+', type=int)
    parser.add_argument("--taskfile", type=str)
    parser.add_argument("--num_steps", type=int, default=5)
    
    parser.add_argument("--fake_data_segment",type=str, default='demo1-seg00')
    parser.add_argument("--fake_data_transform", type=float, nargs=6, metavar=("tx","ty","tz","rx","ry","rz"),
        default=[0,0,0,0,0,0], help="translation=(tx,ty,tz), axis-angle rotation=(rx,ry,rz)")
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--interactive",action="store_true")
    parser.add_argument("--log", type=str, default="", help="")
    
    args = parser.parse_args()

    if args.random_seed is not None: np.random.seed(args.random_seed)

    trajoptpy.SetInteractive(args.interactive)

    if args.log:
        redprint("Writing log to file %s" % args.log)
        Globals.exec_log = task_execution.ExecutionLog(args.log)
        atexit.register(Globals.exec_log.close)
        Globals.exec_log(0, "main.args", args)

    Globals.env = openravepy.Environment()
    Globals.env.StopSimulation()
    Globals.env.Load("robots/pr2-beta-static.zae")
    Globals.robot = Globals.env.GetRobots()[0]

    Globals.actionfile = h5py.File(args.actionfile, 'r')
    
    init_rope_xyz, _ = load_fake_data_segment(Globals.actionfile, args.fake_data_segment, args.fake_data_transform) # this also sets the torso (torso_lift_joint) to the height in the data
    table_height = init_rope_xyz[:,2].mean() - .02
    table_xml = make_table_xml(translation=[1, 0, table_height], extents=[.85, .55, .01])
    Globals.env.LoadData(table_xml)
    if args.elbow_obstacle:
        Globals.env.Load("data/bookshelves.env.xml")

    Globals.sim = ropesim.Simulation(Globals.env, Globals.robot)
    # create rope from rope in data
    rope_nodes = rope_initialization.find_path_through_point_cloud(init_rope_xyz)
    Globals.sim.create(rope_nodes)
    # move arms to the side
    reset_arms_to_side()

    cc = trajoptpy.GetCollisionChecker(Globals.env)
    links_to_exclude = Globals.robot.GetLinks()

    for link in links_to_exclude:
        for rope_link in Globals.env.GetKinBody('rope').GetLinks():
            cc.ExcludeCollisionPair(link, rope_link)
        cc.ExcludeCollisionPair(link, Globals.env.GetKinBody('table').GetLinks()[0])

    if args.animation:
        Globals.viewer = trajoptpy.GetViewer(Globals.env)
        print "move viewer to viewpoint that isn't stupid"
        print "then hit 'p' to continue"
        Globals.viewer.Idle()

    if args.jointopt:
        simulate_demo_fn = simulate_demo_jointopt
    else:
        simulate_demo_fn = simulate_demo

    #####################
    actions = h5py.File(args.actionfile, 'r')

    if args.warpingcost == "regcost":
        feature_fn = regcost_feature_fn
    elif args.warpingcost == "regcost-trajopt":
        feature_fn = regcost_trajopt_feature_fn
    else:
        feature_fn = jointopt_feature_fn

    weights = np.array([-1])
    num_features = 1
    assert weights.shape[0] == num_features, "Dimensions of weights and features don't match. Make sure the right feature is being used"
   
    holdoutfile = h5py.File(args.holdoutfile, 'r')

    save_results = args.resultfile is not None
    
    unique_id = 0
    def get_unique_id():
        global unique_id
        unique_id += 1
        return unique_id-1

    tasks = [] if args.tasks is None else args.tasks
    if args.taskfile is not None:
        file = open(args.taskfile, 'r')
        for line in file.xreadlines():
            tasks.append(int(line[5:-1]))
    if args.i_start != -1 and args.i_end != -1:
        tasks = range(args.i_start, args.i_end)

    def q_value_fn(state, action):
        return np.dot(weights, feature_fn(state, action)) #+ w0
    def value_fn(state):
        state = state[:]
        return max(q_value_fn(state, action) for action in actions)

    num_successes = 0
    num_total = 0

    for i_task, demo_id_rope_nodes in (holdoutfile.iteritems() if not tasks else [(unicode(t),holdoutfile[unicode(t)]) for t in tasks]):
        reset_arms_to_side()

        redprint("Replace rope")
        rope_nodes = demo_id_rope_nodes["rope_nodes"][:]
        replace_rope(rope_nodes)
        Globals.sim.settle()
        if args.animation:
            Globals.viewer.Step()

        if save_results:
            result_file = h5py.File(args.resultfile, 'a')
            if i_task in result_file:
                del result_file[i_task]
            result_file.create_group(i_task)
        
        for i_step in range(args.num_steps):
            print "task %s step %i" % (i_task, i_step)

            reset_arms_to_side()

            redprint("Observe point cloud")
            new_xyz = Globals.sim.observe_cloud()
            state = ("eval_%i"%get_unique_id(), new_xyz)
    

            Globals.sim.observe_cloud()
            if is_knot(Globals.sim.observe_cloud()):
                break;

            redprint("Choosing an action")
            q_values = [q_value_fn(state, action) for action in actions]
            q_values_root = q_values
            rope_tf = get_rope_transforms()

            assert args.lookahead_width>= 1, 'Lookahead branches set to zero will fail to select any action'
            agenda = sorted(zip(q_values, actions), key = lambda v: -v[0])[:args.lookahead_width]
            agenda = [(v, a, rope_tf, a) for (v, a) in agenda] # state is (value, most recent action, rope_transforms, root action)
            best_root_action = None
            for _ in range(args.lookahead_depth):
                expansion_results = []
                for (q, a, tf, r_a) in agenda:
                    set_rope_transforms(tf)                 
                    cur_xyz = Globals.sim.observe_cloud()
                    success, bodypart2trajs = simulate_demo_fn(cur_xyz, Globals.actionfile[a], animate=False)
                    if args.animation:
                        Globals.viewer.Step()
                    result_cloud = Globals.sim.observe_cloud()
                    if is_knot(result_cloud):
                        best_root_action = r_a
                        break
                    expansion_results.append((result_cloud, a, success, get_rope_transforms(), r_a))
                if best_root_action is not None:
                    redprint('Knot Found, stopping search early')
                    break
                agenda = []
                for (cld, incoming_a, success, tf, r_a) in expansion_results:
                    if not success:
                        agenda.append((-np.inf, actions[0], tf, r_a))
                        continue
                    next_state = ("eval_%i"%get_unique_id(), cld)
                    q_values = [(q_value_fn(next_state, action), action, tf, r_a) for action in actions]
                    agenda.extend(q_values)
                agenda.sort(key = lambda v: -v[0])
                agenda = agenda[:args.lookahead_width]                    
                first_root_action = agenda[0][-1]
                if all(r_a == first_root_action for (_, _, _, r_a) in agenda):
                    best_root_action = first_root_action
                    redprint('All best actions have same root, stopping search early')
                    break
            if best_root_action is None:
                best_root_action = agenda[0][-1]
            set_rope_transforms(rope_tf) # reset rope to initial state
            success, trajs = simulate_demo_fn(new_xyz, Globals.actionfile[best_root_action], animate=args.animation)
            set_rope_transforms(get_rope_transforms())
            
            if save_results:
                result_file[i_task].create_group(str(i_step))
                result_file[i_task][str(i_step)]['rope_nodes'] = Globals.sim.rope.GetControlPoints()
                result_file[i_task][str(i_step)]['best_action'] = str(best_root_action)
                trajs_g = result_file[i_task][str(i_step)].create_group('trajs')
                for (i_traj,traj) in enumerate(trajs):
                    traj_g = trajs_g.create_group(str(i_traj))
                    for (bodypart, bodyparttraj) in traj.iteritems():
                        traj_g[str(bodypart)] = bodyparttraj
                result_file[i_task][str(i_step)]['values'] = q_values_root

        if is_knot(Globals.sim.observe_cloud()):
            num_successes += 1
        num_total += 1

        redprint('Successes / Total: ' + str(num_successes) + '/' + str(num_total))
        if save_results:
            result_file.close()
