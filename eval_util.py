# Contains useful functions for evaluating on PR2 rope tying simulation.
# The purpose of this class is to eventually consolidate the various
# instantiations of do_task_eval.py
import sim_util
import openravepy, trajoptpy
import h5py, numpy as np
from rapprentice import math_utils as mu

class EvalStats:
    def __init__(self):
        self.found_feasible_action = False
        self.success = False
        self.feasible = True
        self.misgrasp = False
        self.action_elapsed_time = 0
        self.exec_elapsed_time = 0

def get_specified_tasks(task_list, task_file, i_start, i_end):
    tasks = [] if task_list is None else task_list
    if task_file is not None:
        file = open(task_file, 'r')
        for line in file.xreadlines():
            tasks.append(int(line[5:-1]))
    if i_start != -1 and i_end != -1:
        tasks = range(i_start, i_end)
    return tasks

def get_holdout_items(holdoutfile, tasks):
    if not tasks:
        return holdoutfile.iteritems()
    else:
        return [(unicode(t), holdoutfile[unicode(t)]) for t in tasks]

def save_task_results_init(fname, sim_env, task_index):
    if fname is None:
        return
    result_file = h5py.File(fname, 'a')
    if task_index in result_file:
        del result_file[task_index]
    result_file.create_group(task_index)
    result_file[task_index].create_group('init')
    trans, rots = sim_util.get_rope_transforms(sim_env)
    result_file[task_index]['init']['rope_nodes'] = sim_env.sim.rope.GetControlPoints()
    result_file[task_index]['init']['trans'] = trans
    result_file[task_index]['init']['rots'] = rots
    result_file.close()

def save_task_results_step(fname, sim_env, task_index, step_index, eval_stats, best_root_action, trajs, q_values_root):
    if fname is None:
        return
    result_file = h5py.File(fname, 'a')
    assert(task_index in result_file, "Must call save_task_results_init() before save_task_results_step()")
    
    result_file[task_index].create_group(str(step_index))
    result_file[task_index][str(step_index)]['misgrasp'] = 1 if eval_stats.misgrasp else 0
    result_file[task_index][str(step_index)]['infeasible'] = 1 if not eval_stats.feasible else 0
    result_file[task_index][str(step_index)]['rope_nodes'] = sim_env.sim.rope.GetControlPoints()
    trans, rots = sim_util.get_rope_transforms(sim_env)
    result_file[task_index][str(step_index)]['trans'] = trans
    result_file[task_index][str(step_index)]['rots'] = rots
    result_file[task_index][str(step_index)]['best_action'] = str(best_root_action)
    trajs_g = result_file[task_index][str(step_index)].create_group('trajs')
    for (i_traj,traj) in enumerate(trajs):
        traj_g = trajs_g.create_group(str(i_traj))
    try:
        for (bodypart, bodyparttraj) in traj.iteritems():
            traj_g[str(bodypart)] = bodyparttraj
    except:
        traj_g['arm'] = traj
    result_file[task_index][str(step_index)]['values'] = q_values_root
    result_file[task_index][str(step_index)]['action_time'] = eval_stats.action_elapsed_time
    result_file[task_index][str(step_index)]['exec_time'] = eval_stats.exec_elapsed_time
    result_file.close()

def traj_collisions(sim_env, traj, collision_dist_threshold, n=100):
    """
    Returns the set of collisions. 
    manip = Manipulator or list of indices
    """
    traj_up = mu.interp2d(np.linspace(0,1,n), np.linspace(0,1,len(traj)), traj)
    _ss = openravepy.RobotStateSaver(sim_env.robot)

    cc = trajoptpy.GetCollisionChecker(sim_env.env)

    col_times = []
    for (i,row) in enumerate(traj_up):
        sim_env.robot.SetActiveDOFValues(row)
        col_now = cc.BodyVsAll(sim_env.robot)
        #with util.suppress_stdout():
        #    col_now2 = cc.PlotCollisionGeometry()
        col_now = [cn for cn in col_now if cn.GetDistance() < collision_dist_threshold]
        if col_now:
            print [cn.GetDistance() for cn in col_now]
            col_times.append(i)
            print "trajopt.CollisionChecker: ", len(col_now)
        #print col_now2
        
    return col_times

def traj_is_safe(sim_env, traj, collision_dist_threshold, n=100):
    return traj_collisions(sim_env, traj, collision_dist_threshold, n) == []
