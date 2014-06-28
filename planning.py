from __future__ import division
import openravepy,trajoptpy, numpy as np, json
import util, sim_util
from rapprentice import tps, registration, math_utils as mu
from rapprentice.registration import ThinPlateSpline
import IPython as ipy

def plan_follow_traj(robot, manip_name, ee_link, new_hmats, old_traj, 
                     no_collision_cost_first=False, use_collision_cost=True, start_fixed=False, joint_vel_limits=None,
                     beta_pos = 1000.0, beta_rot = 10.0, gamma = 1000.0):
    orig_dof_inds = robot.GetActiveDOFIndices()
    orig_dof_vals = robot.GetDOFValues()
    
    n_steps = len(new_hmats)
    dof_inds = sim_util.dof_inds_from_name(robot, manip_name)
    assert old_traj.shape[0] == n_steps
    assert old_traj.shape[1] == len(dof_inds)
        
    ee_linkname = ee_link.GetName()
    
    if no_collision_cost_first:
        init_traj, _ = plan_follow_traj(robot, manip_name, ee_link, new_hmats, old_traj, 
                                        no_collision_cost_first=False, use_collision_cost=False, start_fixed=start_fixed, joint_vel_limits=joint_vel_limits,
                                        beta_pos = beta_pos, beta_rot = beta_rot, gamma = gamma)
    else:
        init_traj = old_traj.copy()

    if start_fixed:
        init_traj = np.r_[robot.GetDOFValues(dof_inds)[None,:], init_traj[1:]]
        sim_util.unwrap_in_place(init_traj, dof_inds)
        init_traj += robot.GetDOFValues(dof_inds) - init_traj[0,:]

    request = {
        "basic_info" : {
            "n_steps" : n_steps,
            "manip" : manip_name,
            "start_fixed" : start_fixed
        },
        "costs" : [
        {
            "type" : "joint_vel",
            "params": {"coeffs" : [gamma/n_steps]}
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
                  "dist_pen" : [0.025]  # robot-obstacle distance that penalty kicks in. expands to length n_timesteps
                }
            })
    
    if joint_vel_limits is not None:
        request["constraints"].append(
             {
                "type" : "joint_vel_limits",
                "params": {"vals" : joint_vel_limits,
                           "first_step" : 0,
                           "last_step" : n_steps-1
                           }
              })

    poses = [openravepy.poseFromMatrix(hmat) for hmat in new_hmats]
    for (i_step,pose) in enumerate(poses):
        if start_fixed and i_step == 0:
            continue
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
        orig_dof_vals
        with util.suppress_stdout():
            prob = trajoptpy.ConstructProblem(s, robot.GetEnv()) # create object that stores optimization problem
            result = trajoptpy.OptimizeProblem(prob) # do optimization
    traj = result.GetTraj()
    
    pose_costs = np.sum([cost_val for (cost_type, cost_val) in result.GetCosts() if cost_type == "pose"])
    pose_err = []
    with openravepy.RobotStateSaver(robot):
        for (i_step, hmat) in enumerate(new_hmats):
            if start_fixed and i_step == 0:
                continue
            robot.SetDOFValues(traj[i_step], dof_inds)
            new_hmat = ee_link.GetTransform()
            pose_err.append(openravepy.poseFromMatrix(mu.invertHmat(hmat).dot(new_hmat)))
    pose_err = np.asarray(pose_err)
    pose_costs2 = np.square( (beta_rot/n_steps) * pose_err[:,1:4] ).sum() + np.square( (beta_pos/n_steps) * pose_err[:,4:7] ).sum()

    joint_vel_cost = np.sum([cost_val for (cost_type, cost_val) in result.GetCosts() if cost_type == "joint_vel"])
    joint_vel_err = np.diff(traj, axis=0)
    joint_vel_cost2 = (gamma/n_steps) * np.square(joint_vel_err).sum()
    sim_util.unwrap_in_place(traj, dof_inds)
    joint_vel_err = np.diff(traj, axis=0)
    
    collision_costs = [cost_val for (cost_type, cost_val) in result.GetCosts() if "collision" in cost_type]
    if len(collision_costs) > 0:
        collision_err = np.asarray(collision_costs)
        collision_costs = np.sum(collision_costs)

    obj_value = np.sum([cost_val for (cost_type, cost_val) in result.GetCosts()])
    
    print "{:>15} | {:>10} | {:>10}".format("", "trajopt", "computed")
    print "{:>15} | {:>10}".format("COSTS", "-"*23)
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("joint_vel", joint_vel_cost, joint_vel_cost2)
    if np.isscalar(collision_costs):
        print "{:>15} | {:>10,.4} | {:>10}".format("collision(s)", collision_costs, "-")
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("pose(s)", pose_costs, pose_costs2)
    print "{:>15} | {:>10,.4} | {:>10}".format("total_obj", obj_value, "-")
    print ""

    print "{:>15} | {:>10} | {:>10}".format("", "abs min", "abs max")
    print "{:>15} | {:>10}".format("ERRORS", "-"*23)
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("joint_vel (deg)", np.rad2deg(np.abs(joint_vel_err).min()), np.rad2deg(np.abs(joint_vel_err).max()))
    if np.isscalar(collision_costs):
        print "{:>15} | {:>10,.4} | {:>10,.4}".format("collision(s)", np.abs(-collision_err).min(), np.abs(-collision_err).max())
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("rot pose(s)", np.abs(pose_err[:,1:4]).min(), np.abs(pose_err[:,1:4]).max())
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("trans pose(s)", np.abs(pose_err[:,4:7]).min(), np.abs(pose_err[:,4:7]).max())
    print ""

    # make sure this function doesn't change state of the robot
    assert not np.any(orig_dof_inds - robot.GetActiveDOFIndices())
    assert not np.any(orig_dof_vals - robot.GetDOFValues())
    
    return traj, obj_value, pose_costs

def plan_follow_finger_pts_traj(robot, manip_name, flr2finger_link, flr2finger_rel_pts, flr2finger_pts_traj, old_traj, 
                                no_collision_cost_first=False, use_collision_cost=True, start_fixed=False, joint_vel_limits=None,
                                beta_pos = 10000.0, gamma=1000.0):
    orig_dof_inds = robot.GetActiveDOFIndices()
    orig_dof_vals = robot.GetDOFValues()
    
    n_steps = len(old_traj)
    dof_inds = sim_util.dof_inds_from_name(robot, manip_name)
    assert old_traj.shape[0] == n_steps
    assert old_traj.shape[1] == len(dof_inds)
    
    if no_collision_cost_first:
        init_traj, _ = plan_follow_finger_pts_traj(robot, manip_name, flr2finger_link, flr2finger_rel_pts, flr2finger_pts_traj, old_traj, 
                                                   no_collision_cost_first=False, use_collision_cost=False, start_fixed=start_fixed, joint_vel_limits=joint_vel_limits,
                                                   beta_pos = beta_pos, gamma = gamma)
    else:
        init_traj = old_traj.copy()

    if start_fixed:
        init_traj = np.r_[robot.GetDOFValues(dof_inds)[None,:], init_traj[1:]]
        sim_util.unwrap_in_place(init_traj, dof_inds)
        init_traj += robot.GetDOFValues(dof_inds) - init_traj[0,:]

    request = {
        "basic_info" : {
            "n_steps" : n_steps,
            "manip" : manip_name,
            "start_fixed" : start_fixed
        },
        "costs" : [
        {
            "type" : "joint_vel",
            "params": {"coeffs" : [gamma/n_steps]}
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
                  "dist_pen" : [0.025]  # robot-obstacle distance that penalty kicks in. expands to length n_timesteps
                }
            })
    
    if joint_vel_limits is not None:
        request["constraints"].append(
             {
                "type" : "joint_vel_limits",
                "params": {"vals" : joint_vel_limits,
                           "first_step" : 0,
                           "last_step" : n_steps-1
                           }
              })
    
    for finger_lr, finger_link in flr2finger_link.items():
        finger_linkname = finger_link.GetName()
        finger_rel_pts = flr2finger_rel_pts[finger_lr]
        finger_pts_traj = flr2finger_pts_traj[finger_lr]
        for (i_step, finger_pts) in enumerate(finger_pts_traj):
            if start_fixed and i_step == 0:
                continue
            request["costs"].append(
                {"type":"rel_pts",
                 "params":{
                    "xyzs":finger_pts.tolist(),
                    "rel_xyzs":finger_rel_pts.tolist(),
                    "link":finger_linkname,
                    "timestep":i_step,
                    "pos_coeffs":[beta_pos/n_steps]*4, # there is a coefficient for each of the 4 points
                 }
                })

    s = json.dumps(request)
    with openravepy.RobotStateSaver(robot):
        with util.suppress_stdout():
            prob = trajoptpy.ConstructProblem(s, robot.GetEnv()) # create object that stores optimization problem
            result = trajoptpy.OptimizeProblem(prob) # do optimization
    traj = result.GetTraj()    

    rel_pts_costs = np.sum([cost_val for (cost_type, cost_val) in result.GetCosts() if cost_type == "rel_pts"])
    rel_pts_err = []
    with openravepy.RobotStateSaver(robot):
        for finger_lr, finger_link in flr2finger_link.items():
            finger_linkname = finger_link.GetName()
            finger_rel_pts = flr2finger_rel_pts[finger_lr]
            finger_pts_traj = flr2finger_pts_traj[finger_lr]
            for (i_step, finger_pts) in enumerate(finger_pts_traj):
                if start_fixed and i_step == 0:
                    continue
                robot.SetDOFValues(traj[i_step], dof_inds)
                new_hmat = finger_link.GetTransform()
                rel_pts_err.append(finger_pts - (new_hmat[:3,3][None,:] + finger_rel_pts.dot(new_hmat[:3,:3].T)))
    rel_pts_err = np.concatenate(rel_pts_err, axis=0)
    rel_pts_costs2 = np.square( (beta_pos/n_steps) * rel_pts_err ).sum() # TODO don't square n_steps

    joint_vel_cost = np.sum([cost_val for (cost_type, cost_val) in result.GetCosts() if cost_type == "joint_vel"])
    joint_vel_err = np.diff(traj, axis=0)
    joint_vel_cost2 = (gamma/n_steps) * np.square(joint_vel_err).sum()
    sim_util.unwrap_in_place(traj, dof_inds)
    joint_vel_err = np.diff(traj, axis=0)

    collision_costs = [cost_val for (cost_type, cost_val) in result.GetCosts() if "collision" in cost_type]
    if len(collision_costs) > 0:
        collision_err = np.asarray(collision_costs)
        collision_costs = np.sum(collision_costs)

    obj_value = np.sum([cost_val for (cost_type, cost_val) in result.GetCosts()])
    
    print "{:>15} | {:>10} | {:>10}".format("", "trajopt", "computed")
    print "{:>15} | {:>10}".format("COSTS", "-"*23)
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("joint_vel", joint_vel_cost, joint_vel_cost2)
    if np.isscalar(collision_costs):
        print "{:>15} | {:>10,.4} | {:>10}".format("collision(s)", collision_costs, "-")
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("rel_pts(s)", rel_pts_costs, rel_pts_costs2)
    print "{:>15} | {:>10,.4} | {:>10}".format("total_obj", obj_value, "-")
    print ""

    print "{:>15} | {:>10} | {:>10}".format("", "abs min", "abs max")
    print "{:>15} | {:>10}".format("ERRORS", "-"*23)
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("joint_vel (deg)", np.rad2deg(np.abs(joint_vel_err).min()), np.rad2deg(np.abs(joint_vel_err).max()))
    if np.isscalar(collision_costs):
        print "{:>15} | {:>10,.4} | {:>10,.4}".format("collision(s)", np.abs(-collision_err).min(), np.abs(-collision_err).max())
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("rel_pts(s)", np.abs(rel_pts_err).min(), np.abs(rel_pts_err).max())
    print ""

    # make sure this function doesn't change state of the robot
    assert not np.any(orig_dof_inds - robot.GetActiveDOFIndices())
    assert not np.any(orig_dof_vals - robot.GetDOFValues())
    
    obj_value = np.sum([cost_val for (cost_type, cost_val) in result.GetCosts()])
    return traj, obj_value, rel_pts_costs

def tps_fit3_ext(x_na, y_ng, bend_coef, rot_coef, wt_n):
    if wt_n is None: wt_n = np.ones(len(x_na))
    n,d = x_na.shape
    
    K_nn = tps.tps_kernel_matrix(x_na)
    Q = np.c_[np.ones((n,1)), x_na, K_nn]
    rot_coefs = np.ones(d) * rot_coef if np.isscalar(rot_coef) else rot_coef

    A = np.r_[np.zeros((d+1,d+1)), np.c_[np.ones((n,1)), x_na]].T
    n_cnts = A.shape[0]
    _u,_s,_vh = np.linalg.svd(A.T)
    N = _u[:,n_cnts:]
    
    solve_dim_separately = not np.isscalar(bend_coef) or (wt_n.ndim > 1 and wt_n.shape[1] > 1)
    
    if solve_dim_separately:
        bend_coefs = np.ones(d) * bend_coef if np.isscalar(bend_coef) else bend_coef
        if wt_n.ndim == 1:
            wt_n = wt_n[:,None]
        if wt_n.shape[1] == 1:
            wt_n = np.tile(wt_n, (1,d))
        z = np.empty((n,d))
        for i in range(d):
            WQ = wt_n[:,i][:,None] * Q
            QWQ = Q.T.dot(WQ)
            H = QWQ
            H[d+1:,d+1:] += bend_coefs[i] * K_nn
            H[1:d+1, 1:d+1] += np.diag(rot_coefs)
             
            f = -WQ.T.dot(y_ng[:,i])
            f[1+i] -= rot_coefs[i]
             
            z[:,i] = np.linalg.solve(N.T.dot(H.dot(N)), -N.T.dot(f))
    else:
        WQ = wt_n[:,None] * Q
        QWQ = Q.T.dot(WQ)
        H = QWQ
        H[d+1:,d+1:] += bend_coef * K_nn
        H[1:d+1, 1:d+1] += np.diag(rot_coefs)
        
        f = -WQ.T.dot(y_ng)
        f[1:d+1,0:d] -= np.diag(rot_coefs)
        
        z = np.linalg.solve(N.T.dot(H.dot(N)), -N.T.dot(f))
    
    return N, z

def tps_obj(f, x_na, y_ng, bend_coef, rot_coef, wt_n):
    # expand these
    _,d = x_na.shape
    bend_coefs = np.ones(d) * bend_coef if np.isscalar(bend_coef) else bend_coef
    rot_coefs = np.ones(d) * rot_coef if np.isscalar(rot_coef) else rot_coef
    if wt_n is None: wt_n = np.ones(len(x_na))
    if wt_n.ndim == 1:
        wt_n = wt_n[:,None]
    if wt_n.shape[1] == 1:
        wt_n = np.tile(wt_n, (1,d))
    
    K_nn = tps.tps_kernel_matrix(x_na)
    _,d = x_na.shape
    cost = 0
    # matching cost
    cost += np.linalg.norm((f.transform_points(x_na) - y_ng) * np.sqrt(wt_n))**2
    # same as (np.square(np.apply_along_axis(np.linalg.norm, 1, f.transform_points(x_na) - y_ng)) * wt_n).sum()
    # bending cost
    cost += np.trace(np.diag(bend_coefs).dot(f.w_ng.T.dot(K_nn.dot(f.w_ng))))
    # rotation cost
    cost += np.trace((f.lin_ag - np.eye(d)).T.dot(np.diag(rot_coefs).dot((f.lin_ag - np.eye(d)))))
#     # constants
#     cost -= np.linalg.norm(y_ng * np.sqrt(wt_n))**2
#     cost -= np.trace(np.diag(rot_coefs))
    return cost

def fit_ThinPlateSpline(x_na, y_ng, bend_coef=.1, rot_coef = 1e-5, wt_n=None):
    env = openravepy.Environment()
    env.StopSimulation()
    env.Load("./data/misc/pr2-beta-static-decomposed-shoulder.zae")
    robot = env.GetRobots()[0]
    alpha = 1.0
    
    (n,d) = x_na.shape

    # expand these
    bend_coefs = np.ones(d) * bend_coef if np.isscalar(bend_coef) else bend_coef
    rot_coefs = np.ones(d) * rot_coef if np.isscalar(rot_coef) else rot_coef
    if wt_n is None: wt_n = np.ones(len(x_na))
    if wt_n.ndim == 1:
        wt_n = wt_n[:,None]
    if wt_n.shape[1] == 1:
        wt_n = np.tile(wt_n, (1,d))
    
    N, z = tps_fit3_ext(x_na, y_ng, bend_coefs, rot_coefs, wt_n)
    
    n_steps = 1 #dummy

    request = {
        "basic_info" : {
            "n_steps" : n_steps,
            "m_ext" : n, 
            "n_ext" : d,
            "manip" : "leftarm", #dummy
            "start_fixed" : False
        },
        "costs" : [
        {
            "type" : "tps",
            "name" : "tps",
            "params" : {"x_na" : [row.tolist() for row in x_na],
                        "y_ng" : [row.tolist() for row in y_ng],
                        "bend_coefs" : bend_coefs.tolist(),
                        "rot_coefs" : rot_coefs.tolist(),
                        "wt_n" : [row.tolist() for row in wt_n],
                        "N" : [row.tolist() for row in N],
                        "alpha" : alpha,
            }
        },
        ],
        "constraints" : [
        ],
    }
    request['init_info'] = {
                                "type":"given_traj",
                                "data":[robot.GetDOFValues(robot.GetManipulator("leftarm").GetArmIndices()).tolist()],
                                "data_ext":[row.tolist() for row in z]
                           }
    
    s = json.dumps(request)
    with openravepy.RobotStateSaver(robot):
        prob = trajoptpy.ConstructProblem(s, robot.GetEnv()) # create object that stores optimization problem
        result = trajoptpy.OptimizeProblem(prob) # do optimization
    traj = result.GetTraj()
    theta = N.dot(result.GetExt())
    f = ThinPlateSpline()
    f.trans_g = theta[0,:];
    f.lin_ag = theta[1:d+1,:];
    f.w_ng = theta[d+1:];
    f.x_na = x_na
    
    tps_cost = 0
    for (cost_type, cost_val) in result.GetCosts():
        if cost_type == "tps":
            tps_cost += cost_val

    def calcForwardNumJac(f, x, epsilon):
        y = f(x)
        out = np.zeros((len(y), len(x)))
        xpert = x.copy()
        for i in range(len(x)):
            xpert[i] = x[i] + epsilon
            ypert = f(xpert)
            out[:,i] = (ypert - y) / epsilon
            xpert[i] = x[i]
        return out
    
    from rapprentice.tps_registration import tile
    dof_inds = robot.GetManipulator("leftarm").GetArmIndices()
    dof_vals = np.array([ 1.832, -0.332,  1.011, -1.437,  1.1  , -2.106,  3.074])
    z = result.GetExt()
    z_vals = z.T.flatten()
    dof_z_vals = np.r_[dof_vals, z_vals]
    src_pts = np.random.random((8,3))
    rel_pts = np.random.random((8,3))
    
    def get_transform(dof_vals):
        with openravepy.RobotStateSaver(robot):
            robot.SetDOFValues(dof_vals, dof_inds)
            hmat = robot.GetManipulator("leftarm").GetTransform()
        return hmat

    def get_trans(dof_vals):
        return get_transform(dof_vals)[:3,3]
    
    def get_rot_T_flatten(dof_vals):
        return get_transform(dof_vals)[:3,:3].flatten()
    
    def rel_pts_err(dof_z_vals):
        n_dof = 7
        dof_vals = dof_z_vals[:n_dof]
        z_vals = dof_z_vals[n_dof:]
        
        dof_inds = robot.GetManipulator("leftarm").GetArmIndices()
        with openravepy.RobotStateSaver(robot):
            robot.SetDOFValues(dof_vals, dof_inds)
            cur_hmat = robot.GetManipulator("leftarm").GetTransform()
        
        theta = N.dot(z_vals.reshape((3,-1)).T)
        f = ThinPlateSpline()
        f.trans_g = theta[0,:];
        f.lin_ag = theta[1:d+1,:];
        f.w_ng = theta[d+1:];
        f.x_na = x_na
        
        warped_pts = f.transform_points(src_pts)

        err = (warped_pts - (cur_hmat[:3,3][None,:] + rel_pts.dot(cur_hmat[:3,:3].T))).T.flatten()
        return err
    J_err_num = calcForwardNumJac(rel_pts_err, dof_z_vals, 1e-10)

    J_trans = calcForwardNumJac(get_trans, dof_vals, 1e-10)
    J_rot_T = calcForwardNumJac(get_rot_T_flatten, dof_vals, 1e-10)
    J_tf = np.repeat(J_trans, len(rel_pts), axis=0) + np.r_[rel_pts.dot(J_rot_T[0:3,:]), rel_pts.dot(J_rot_T[3:6,:]), rel_pts.dot(J_rot_T[6:9,:])]
    Q = np.c_[np.ones((len(src_pts),1)), src_pts, tps.tps_kernel_matrix2(src_pts, x_na)]
    J_tps = tile(Q.dot(N), np.eye(3))
    J_err = np.c_[-J_tf, J_tps]
    
    print np.linalg.norm(J_err_num - J_err)
    
    ipy.embed()

    return f, tps_cost

if __name__ == "__main__":
    x_na = np.load("x_na.npy")
    y_ng = np.load("y_ng.npy")
    bend_coef = np.array([0.00015, 0.00015, 0.0015])
    rot_coef = np.r_[1e-4, 1e-4, 1e-1]
    f, tps_cost = fit_ThinPlateSpline(x_na, y_ng, bend_coef=bend_coef, rot_coef=rot_coef)


def joint_fit_tps_follow_finger_pts_traj(robot, manip_name, flr2finger_link, flr2finger_rel_pts, flr2old_finger_pts_traj, old_traj, 
                                         x_na, y_ng, bend_coef, rot_coef, wt_n, old_N_z = None, closing_pts = None,
                                         no_collision_cost_first=False, use_collision_cost=True, start_fixed=False, joint_vel_limits=None,
                                         alpha = 1.0, beta_pos = 10000.0, gamma = 1000.0):
    orig_dof_inds = robot.GetActiveDOFIndices()
    orig_dof_vals = robot.GetDOFValues()
    
    n_steps = len(old_traj)
    dof_inds = sim_util.dof_inds_from_name(robot, manip_name)
    assert old_traj.shape[0] == n_steps
    assert old_traj.shape[1] == len(dof_inds)

    (n,d) = x_na.shape

    # expand these
    bend_coefs = np.ones(d) * bend_coef if np.isscalar(bend_coef) else bend_coef
    rot_coefs = np.ones(d) * rot_coef if np.isscalar(rot_coef) else rot_coef
    if wt_n is None: wt_n = np.ones(len(x_na))
    if wt_n.ndim == 1:
        wt_n = wt_n[:,None]
    if wt_n.shape[1] == 1:
        wt_n = np.tile(wt_n, (1,d))
    
    if no_collision_cost_first:
        init_traj, _, (N, init_z) , _, _ = joint_fit_tps_follow_finger_pts_traj(robot, manip_name, flr2finger_link, flr2finger_rel_pts, flr2old_finger_pts_traj, old_traj, 
                                                                                x_na, y_ng, bend_coefs, rot_coefs, wt_n, old_N_z=old_N_z, closing_pts=closing_pts, 
                                                                                no_collision_cost_first=False, use_collision_cost=False, start_fixed=start_fixed, joint_vel_limits=joint_vel_limits,
                                                                                alpha = alpha, beta_pos = beta_pos, gamma = gamma)
    else:
        init_traj = old_traj.copy() # is copy needed?
        if old_N_z is None:
            N, init_z = tps_fit3_ext(x_na, y_ng, bend_coefs, rot_coefs, wt_n)
        else:
            N, init_z = old_N_z

    if start_fixed:
        init_traj = np.r_[robot.GetDOFValues(dof_inds)[None,:], init_traj[1:]]
        sim_util.unwrap_in_place(init_traj, dof_inds)
        init_traj += robot.GetDOFValues(dof_inds) - init_traj[0,:]

    request = {
        "basic_info" : {
            "n_steps" : n_steps,
            "m_ext" : n, 
            "n_ext" : d,
            "manip" : manip_name,
            "start_fixed" : start_fixed
        },
        "costs" : [
        {
            "type" : "joint_vel",
            "params": {"coeffs" : [gamma/n_steps]}
        },
        {
            "type" : "tps",
            "name" : "tps",
            "params" : {"x_na" : [row.tolist() for row in x_na],
                        "y_ng" : [row.tolist() for row in y_ng],
                        "bend_coefs" : bend_coefs.tolist(),
                        "rot_coefs" : rot_coefs.tolist(),
                        "wt_n" : [row.tolist() for row in wt_n],
                        "N" : [row.tolist() for row in N],
                        "alpha" : alpha,
            }
        }
        ],
        "constraints" : [
        ],
        "init_info" : {
            "type":"given_traj",
            "data":[x.tolist() for x in init_traj],
            "data_ext":[row.tolist() for row in init_z]
        }
    }
    
    if use_collision_cost:
        request["costs"].append(
            {
                "type" : "collision",
                "params" : {
                  "continuous" : True,
                  "coeffs" : [1000],  # penalty coefficients. list of length one is automatically expanded to a list of length n_timesteps
                  "dist_pen" : [0.025]  # robot-obstacle distance that penalty kicks in. expands to length n_timesteps
                }
            })
    
    if joint_vel_limits is not None:
        request["constraints"].append(
             {
                "type" : "joint_vel_limits",
                "params": {"vals" : joint_vel_limits,
                           "first_step" : 0,
                           "last_step" : n_steps-1
                           }
              })

    if closing_pts is not None:
        request["costs"].append(
            {
                "type":"tps_jac_orth",
                "params":  {
                            "tps_cost_name":"tps",
                            "pts":closing_pts.tolist(),
                            "coeffs":[10.0]*len(closing_pts),
                            }
            })
    
    for finger_lr, finger_link in flr2finger_link.items():
        finger_linkname = finger_link.GetName()
        finger_rel_pts = flr2finger_rel_pts[finger_lr]
        old_finger_pts_traj = flr2old_finger_pts_traj[finger_lr]
        for (i_step, old_finger_pts) in enumerate(old_finger_pts_traj):
            if start_fixed and i_step == 0:
                continue
            request["costs"].append(
                {"type":"tps_rel_pts",
                 "params":{
                    "tps_cost_name":"tps",
                    "src_xyzs":old_finger_pts.tolist(),
                    "rel_xyzs":finger_rel_pts.tolist(),
                    "link":finger_linkname,
                    "timestep":i_step,
                    "pos_coeffs":[beta_pos/n_steps]*4,
                 }
                })

    s = json.dumps(request)
    with openravepy.RobotStateSaver(robot):
        with util.suppress_stdout():
            prob = trajoptpy.ConstructProblem(s, robot.GetEnv()) # create object that stores optimization problem
            result = trajoptpy.OptimizeProblem(prob) # do optimization
    traj = result.GetTraj()
    z = result.GetExt()
    theta = N.dot(z)
    f = ThinPlateSpline()
    f.trans_g = theta[0,:];
    f.lin_ag = theta[1:d+1,:];
    f.w_ng = theta[d+1:];
    f.x_na = x_na
    
    tps_rel_pts_costs = np.sum([cost_val for (cost_type, cost_val) in result.GetCosts() if cost_type == "tps_rel_pts"])
    tps_rel_pts_err = []
    with openravepy.RobotStateSaver(robot):
        for finger_lr, finger_link in flr2finger_link.items():
            finger_linkname = finger_link.GetName()
            finger_rel_pts = flr2finger_rel_pts[finger_lr]
            old_finger_pts_traj = flr2old_finger_pts_traj[finger_lr]
            for (i_step, old_finger_pts) in enumerate(old_finger_pts_traj):
                if start_fixed and i_step == 0:
                    continue
                robot.SetDOFValues(traj[i_step], dof_inds)
                new_hmat = finger_link.GetTransform()
                tps_rel_pts_err.append(f.transform_points(old_finger_pts) - (new_hmat[:3,3][None,:] + finger_rel_pts.dot(new_hmat[:3,:3].T)))
    tps_rel_pts_err = np.concatenate(tps_rel_pts_err, axis=0)
    tps_rel_pts_costs2 = np.square( (beta_pos/n_steps) * tps_rel_pts_err ).sum() # TODO don't square n_steps

    tps_cost = np.sum([cost_val for (cost_type, cost_val) in result.GetCosts() if cost_type == "tps"])
    tps_cost2 = alpha * tps_obj(f, x_na, y_ng, bend_coefs, rot_coefs, wt_n)
    matching_err = f.transform_points(x_na) - y_ng
    
    joint_vel_cost = np.sum([cost_val for (cost_type, cost_val) in result.GetCosts() if cost_type == "joint_vel"])
    joint_vel_err = np.diff(traj, axis=0)
    joint_vel_cost2 = (gamma/n_steps) * np.square(joint_vel_err).sum()
    sim_util.unwrap_in_place(traj, dof_inds)
    joint_vel_err = np.diff(traj, axis=0)

    collision_costs = [cost_val for (cost_type, cost_val) in result.GetCosts() if "collision" in cost_type]
    if len(collision_costs) > 0:
        collision_err = np.asarray(collision_costs)
        collision_costs = np.sum(collision_costs)

    tps_jac_orth_cost = [cost_val for (cost_type, cost_val) in result.GetCosts() if "tps_jac_orth" in cost_type]
    if len(tps_jac_orth_cost) > 0:
        tps_jac_orth_cost = np.sum(tps_jac_orth_cost)
        f_jacs = f.compute_jacobian(closing_pts)
        tps_jac_orth_err = []
        for jac in f_jacs:
            tps_jac_orth_err.extend((jac.dot(jac.T) - np.eye(3)).flatten())
        tps_jac_orth_err = np.asarray(tps_jac_orth_err)
        tps_jac_orth_cost2 = np.square( 10.0 * tps_jac_orth_err ).sum()

    obj_value = np.sum([cost_val for (cost_type, cost_val) in result.GetCosts()])
    
    print "{:>15} | {:>10} | {:>10}".format("", "trajopt", "computed")
    print "{:>15} | {:>10}".format("COSTS", "-"*23)
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("joint_vel", joint_vel_cost, joint_vel_cost2)
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("tps", tps_cost, tps_cost2)
    if np.isscalar(collision_costs):
        print "{:>15} | {:>10,.4} | {:>10}".format("collision(s)", collision_costs, "-")
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("tps_rel_pts(s)", tps_rel_pts_costs, tps_rel_pts_costs2)
    if np.isscalar(tps_jac_orth_cost):
        print "{:>15} | {:>10,.4} | {:>10,.4}".format("tps_jac_orth", tps_jac_orth_cost, tps_jac_orth_cost2)
    print "{:>15} | {:>10,.4} | {:>10}".format("total_obj", obj_value, "-")
    print ""

    print "{:>15} | {:>10} | {:>10}".format("", "abs min", "abs max")
    print "{:>15} | {:>10}".format("ERRORS", "-"*23)
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("joint_vel (deg)", np.rad2deg(np.abs(joint_vel_err).min()), np.rad2deg(np.abs(joint_vel_err).max()))
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("tps (matching)", np.abs(matching_err).min(), np.abs(matching_err).max())
    if np.isscalar(collision_costs):
        print "{:>15} | {:>10,.4} | {:>10,.4}".format("collision(s)", np.abs(-collision_err).min(), np.abs(-collision_err).max())
    print "{:>15} | {:>10,.4} | {:>10,.4}".format("tps_rel_pts(s)", np.abs(tps_rel_pts_err).min(), np.abs(tps_rel_pts_err).max())
    if np.isscalar(tps_jac_orth_cost):
        print "{:>15} | {:>10,.4} | {:>10,.4}".format("tps_jac_orth", np.abs(tps_jac_orth_err).min(), np.abs(tps_jac_orth_err).max())
    print ""

    # make sure this function doesn't change state of the robot
    assert not np.any(orig_dof_inds - robot.GetActiveDOFIndices())
    assert not np.any(orig_dof_vals - robot.GetDOFValues())
    
    return traj, f, (N, z), obj_value, tps_rel_pts_costs, tps_cost
