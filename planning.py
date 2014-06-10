from __future__ import division
import openravepy,trajoptpy, numpy as np, json
import util
from rapprentice import tps, math_utils as mu
from rapprentice.registration import ThinPlateSpline
import IPython as ipy

def plan_follow_traj(robot, manip_name, ee_link, new_hmats, old_traj, beta_pos = 1000.0, beta_rot = 10., no_collision_cost_first=False, use_collision_cost=True):
    orig_dof_inds = robot.GetActiveDOFIndices()
    orig_dof_vals = robot.GetDOFValues()
        
    n_steps = len(new_hmats)
    assert old_traj.shape[0] == n_steps
    assert old_traj.shape[1] == 7
    
    arm_inds  = robot.GetManipulator(manip_name).GetArmIndices()

    ee_linkname = ee_link.GetName()
    
    if no_collision_cost_first:
        init_traj, _ = plan_follow_traj(robot, manip_name, ee_link, new_hmats, old_traj, beta_pos = beta_pos, beta_rot = beta_rot, no_collision_cost_first=False, use_collision_cost=False)
    else:
        init_traj = old_traj.copy()

    request = {
        "basic_info" : {
            "n_steps" : n_steps,
            "manip" : manip_name,
            "start_fixed" : False
        },
        "costs" : [
        {
            "type" : "joint_vel",
            "params": {"coeffs" : [1./float(n_steps)]}
        },            
        ],
        "constraints" : [
        ],
        "init_info" : {
            "type":"given_traj",
            "data":[x.tolist() for x in init_traj]
        }
    }
    
    if use_collision_cost:
        request["costs"].append(
            {
                "type" : "collision",
                "params" : {
                  "continuous" : True,
                  "coeffs" : [1000],  # penalty coefficients. list of length one is automatically expanded to a list of length n_timesteps
                  "dist_pen" : [0.01]  # robot-obstacle distance that penalty kicks in. expands to length n_timesteps
                }
            })

    poses = [openravepy.poseFromMatrix(hmat) for hmat in new_hmats]
    for (i_step,pose) in enumerate(poses):
        request["costs"].append(
            {"type":"pose",
             "params":{
                "xyz":pose[4:7].tolist(),
                "wxyz":pose[0:4].tolist(),
                "link":ee_linkname,
                "timestep":i_step,
                "pos_coeffs":[beta_pos/n_steps]*3,
                "rot_coeffs":[beta_rot/n_steps]*3
             }
            })

    s = json.dumps(request)
    with openravepy.RobotStateSaver(robot):
        with util.suppress_stdout():
            prob = trajoptpy.ConstructProblem(s, robot.GetEnv()) # create object that stores optimization problem
            result = trajoptpy.OptimizeProblem(prob) # do optimization
    print result.GetCosts()
    print result.GetConstraints()
    traj = result.GetTraj()    

    pose_costs = 0
    for (cost_type, cost_val) in result.GetCosts():
        if cost_type == 'pose':
            pose_costs += cost_val

    pose_costs2 = 0
    with openravepy.RobotStateSaver(robot):
        for (i_step, hmat) in enumerate(new_hmats):
            robot.SetDOFValues(traj[i_step], arm_inds)
            new_hmat = ee_link.GetTransform()
            pose_err = openravepy.poseFromMatrix(mu.invertHmat(hmat).dot(new_hmat))
            pose_costs2 += ((beta_rot/n_steps) * np.linalg.norm(pose_err[1:4]))**2 + ((beta_pos/n_steps) * np.linalg.norm(pose_err[4:7]))**2
    print "pose_costs", pose_costs, pose_costs2

    print "planned trajectory for %s. total pose error: %.3f."%(manip_name, pose_costs)

    # make sure this function doesn't change state of the robot
    assert not np.any(orig_dof_inds - robot.GetActiveDOFIndices())
    assert not np.any(orig_dof_vals - robot.GetDOFValues())
    
    return traj, pose_costs


def prepare_tps_fit3(x_na, y_ng, bend_coef, rot_coef, wt_n):
    if wt_n is None: wt_n = np.ones(len(x_na))
    n,d = x_na.shape

    K_nn = tps.tps_kernel_matrix(x_na)
    Q = np.c_[np.ones((n,1)), x_na, K_nn]
    WQ = wt_n[:,None] * Q
    QWQ = Q.T.dot(WQ)
    H = QWQ
    H[d+1:,d+1:] += bend_coef * K_nn
    rot_coefs = np.ones(d) * rot_coef if np.isscalar(rot_coef) else rot_coef
    H[1:d+1, 1:d+1] += np.diag(rot_coefs)
    f = -2.0*WQ.T.dot(y_ng)
    f[1:d+1,0:d] -= np.diag(rot_coefs)
    
    A = np.r_[np.zeros((d+1,d+1)), np.c_[np.ones((n,1)), x_na]].T

    n_vars = H.shape[0]
    assert H.shape[1] == n_vars
    assert f.shape[0] == n_vars
    assert A.shape[1] == n_vars
    n_cnts = A.shape[0]
    
    _u,_s,_vh = np.linalg.svd(A.T)
    N = _u[:,n_cnts:]
    
    z = np.linalg.solve(2.0*N.T.dot(H.dot(N)), -N.T.dot(f))
    
    return H, f, A, N, z, wt_n, rot_coefs


def joint_fit_tps_follow_traj(robot, manip_name, ee_links, fn, old_hmats_list, old_trajs, x_na, y_ng, alpha = 1., beta_pos = 1000.0, beta_rot = 10, bend_coef=.1, rot_coef = 1e-5, wt_n=None):
    """
    The order of dof indices in hmats and traj should be the same as especified by manip_name

    Note: returns tps_cost = (alpha / # correspondence points) * orig_tps_cost,
                  pose_costs = (beta / # trajectory points) * orig_pose_costs
    """
    orig_dof_inds = robot.GetActiveDOFIndices()
    orig_dof_vals = robot.GetDOFValues()
    
    (n,d) = x_na.shape
    H, f, A, N, z, wt_n, rot_coefs = prepare_tps_fit3(x_na, y_ng, bend_coef, rot_coef, wt_n)
    
    n_steps = len(old_hmats_list[0])
    
    arm_inds = []
    for name in manip_name.split('+'):
        arm_inds.append(robot.GetManipulator(name).GetArmIndices())
    arm_inds = np.concatenate(arm_inds)
    
    request = {
        "basic_info" : {
            "n_steps" : n_steps,
            "m_ext" : n, 
            "n_ext" : d,
            "manip" : manip_name,
            "start_fixed" : False
        },
        "costs" : [
#         {
#             "type" : "joint_vel",
#             "params": {"coeffs" : [n_steps/5.]}
#         },
        {
            "type" : "tps",
            "name" : "tps",
            "params" : {"H" : [row.tolist() for row in H],
                        "f" : [row.tolist() for row in f],
                        "x_na" : [row.tolist() for row in x_na],
                        "N" : [row.tolist() for row in N],
                        "y_ng" : [row.tolist() for row in y_ng],
                        "wt_n" : wt_n.tolist(),
                        "rot_coef" : rot_coefs.tolist(),
                        "alpha" : alpha,
                        "lambda" : bend_coef,
            }
        },
        {
            "type" : "collision",
            "params" : {
              "continuous" : True,
              "coeffs" : [1000],  # penalty coefficients. list of length one is automatically expanded to a list of length n_timesteps
              "dist_pen" : [0.01]  # robot-obstacle distance that penalty kicks in. expands to length n_timesteps
            }
        },
        ],
        "constraints" : [
        ],
    }
    
    init_trajs = []
    for ee_link, old_hmats, old_traj in zip(ee_links, old_hmats_list, old_trajs):
        assert len(old_hmats) == n_steps
        assert old_traj.shape[0] == n_steps
        assert old_traj.shape[1] == 7

        ee_linkname = ee_link.GetName()
    
        init_traj = old_traj.copy()
        init_trajs.append(init_traj)
        
        poses = [openravepy.poseFromMatrix(hmat) for hmat in old_hmats]
        for (i_step,pose) in enumerate(poses):
#             if i_step != 0:
            request["costs"].append(
                {"type":"tps_pose",
                 "params":{
                    "x_na":[row.tolist() for row in x_na],
                    "N" : [row.tolist() for row in N],
                    "xyz":pose[4:7].tolist(),
                    "wxyz":pose[0:4].tolist(),
                    "link":ee_linkname,
                    "timestep":i_step,
                    "pos_coeffs":[beta_pos/n_steps]*3,
                    "rot_coeffs":[beta_rot/n_steps]*3
                 }
                })
#         for (i_step,hmat) in zip([0], fn.transform_hmats(np.array([old_hmats[0]]))):
#             pose = openravepy.poseFromMatrix(hmat)
#             request['constraints'].append(
#                 {
#                     "type" : "pose", 
#                     "params" : {
#                         "xyz":pose[4:7].tolist(),
#                         "wxyz":pose[0:4].tolist(),
#                         "link": ee_linkname,
#                         "timestep" : i_step
#                     }
#                 })
    
    request['init_info'] = {
                                "type":"given_traj",
                                "data":[x.tolist() for x in np.concatenate(init_trajs, axis=1)],
                                "data_ext":[row.tolist() for row in z]
                           }
    
    s = json.dumps(request)
    with openravepy.RobotStateSaver(robot):
        with util.suppress_stdout():
            prob = trajoptpy.ConstructProblem(s, robot.GetEnv()) # create object that stores optimization problem
            result = trajoptpy.OptimizeProblem(prob) # do optimization
    print result.GetCosts()
    print result.GetConstraints()
    traj = result.GetTraj()
    theta = N.dot(result.GetExt())
    f = ThinPlateSpline()
    f.trans_g = theta[0,:];
    f.lin_ag = theta[1:d+1,:];
    f.w_ng = theta[d+1:];
    f.x_na = x_na
    
    tps_cost = 0
    pose_costs = 0
    for (cost_type, cost_val) in result.GetCosts():
        if cost_type == "tps":
            tps_cost += cost_val
        elif cost_type == "tps_pose":
            pose_costs += cost_val
    
    w_sqrt = np.sqrt(wt_n)
    f_x_na = f.transform_points(x_na)
    K = tps.tps_kernel_matrix(x_na)
    tps_cost2 = (alpha/x_na.shape[0]) * (np.linalg.norm((f_x_na - y_ng) * np.repeat(w_sqrt[:,None],3,1))**2 \
        + bend_coef * (f.w_ng.T.dot(K.dot(f.w_ng))).trace() \
        + (f.lin_ag.T.dot(np.diag(rot_coefs).dot(f.lin_ag))).trace() \
        - np.diag(rot_coefs).dot(f.lin_ag.T).trace())
    print "tps_cost", tps_cost, tps_cost2

    pose_costs2 = 0
    with openravepy.RobotStateSaver(robot):
        for ee_link, old_hmats in zip(ee_links, old_hmats_list):
            for (i_step, old_hmat) in enumerate(old_hmats):
                robot.SetDOFValues(traj[i_step], arm_inds)
                cur_hmat = ee_link.GetTransform()
                warped_src_hmat = f.transform_hmats(old_hmat[None,:,:])[0]
                pose_err = openravepy.poseFromMatrix(mu.invertHmat(warped_src_hmat).dot(cur_hmat))
                pose_costs2 += ((beta_rot/n_steps) * np.linalg.norm(pose_err[1:4]))**2 + ((beta_pos/n_steps) * np.linalg.norm(pose_err[4:7]))**2
    print "pose_costs", pose_costs, pose_costs2

    print "planned trajectory for %s. tps error: %.3f. total pose error: %.3f."%(manip_name, tps_cost, pose_costs)
    
    # make sure this function doesn't change state of the robot
    assert not np.any(orig_dof_inds - robot.GetActiveDOFIndices())
    assert not np.any(orig_dof_vals - robot.GetDOFValues())

    # f is the warping function
    return traj, f, tps_cost, pose_costs
