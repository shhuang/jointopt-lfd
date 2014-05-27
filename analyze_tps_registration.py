import matplotlib.pyplot as plt
import numpy as np
import argparse, h5py
import eval_util
from rapprentice import registration, tps_registration, clouds, plotting_plt
import IPython as ipy

N_ITER_CHEAP = 14 # do_task_eval uses 8 # TODO
EM_ITER_CHEAP = 1
DS_SIZE = 0.025

def plot_tps_registrations_proj_2d(x_nds, y_md, fs, res, x_colors, y_color):
    # set interactive
    plt.ion()
    
    k = len(x_nds) # number of demonstrations
    
    fig, axes = plt.subplots(num='2d projection plot', nrows=k, ncols=4)

    for i, (x_nd, f, x_color) in enumerate(zip(x_nds, fs, x_colors)):
        axes[i,0].clear()
        axes[i,0].set_aspect('equal')
        axes[i,0].scatter(x_nd[:,0], x_nd[:,1], c=x_color, edgecolors=x_color, marker=',', s=5)
    
        grid_means = .5 * (x_nd.max(axis=0) + x_nd.min(axis=0))
        grid_mins = grid_means - (x_nd.max(axis=0) - x_nd.min(axis=0))
        grid_maxs = grid_means + (x_nd.max(axis=0) - x_nd.min(axis=0))
        x_median = np.median(x_nd, axis=0)
        plt.sca(axes[i,0])
        plotting_plt.plot_warped_grid_proj_2d(lambda xyz: xyz, grid_mins[:2], grid_maxs[:2], z=x_median[2], xres=res[0], yres=res[1], draw=False)
    
        axes[i,1].clear()
        axes[i,1].set_aspect('equal')
        axes[i,1].scatter(y_md[:,0], y_md[:,1], c=y_color, edgecolors=y_color, marker=',', s=5)
    
        axes[i,2].clear()
        axes[i,2].set_aspect('equal')
        xwarped_nd = f.transform_points(x_nd)
        axes[i,2].scatter(xwarped_nd[:,0], xwarped_nd[:,1], c=x_color, edgecolors=x_color, marker=',', s=5)
        plt.sca(axes[i,2])
        plotting_plt.plot_warped_grid_proj_2d(f.transform_points, grid_mins[:2], grid_maxs[:2], z=x_median[2], xres=res[0], yres=res[1], draw=False)

        axes[i,3].clear()
        axes[i,3].axis('off')
        axes[i,3].text(0.5, 0.5, 'cost %.2f\n reg_cost %.2f'%(f._cost, registration.tps_reg_cost(f)),
            horizontalalignment='center',
            verticalalignment='center')

    fig.tight_layout()
    
    plt.draw()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('actionfile', type=str)
    parser.add_argument('cloudfile', type=str, help="result file containing the encountered clouds")
    parser.add_argument("--tasks", type=int, nargs='*', metavar="i_task")
    parser.add_argument("--taskfile", type=str)
    parser.add_argument("--i_start", type=int, default=-1, metavar="i_task")
    parser.add_argument("--i_end", type=int, default=-1, metavar="i_task")

    args = parser.parse_args()
    
    actionfile = h5py.File(args.actionfile, 'r')
    cloudfile = h5py.File(args.cloudfile, 'r')
    cloud_items = eval_util.get_holdout_items(cloudfile, args.tasks, args.taskfile, args.i_start, args.i_end)

    for i_task, _ in cloud_items:
        for i_step in range(len(cloudfile[i_task]) - (1 if 'init' in cloudfile[i_task] else 0)):
            print "task %s step %i" % (i_task, i_step)
            task_index = str(i_task)
            step_index = str(i_step)

            sel_old_cloud = cloudfile[task_index][step_index]['demo_cloud'][()]
            sel_old_cloud_ds = cloudfile[task_index][step_index]['demo_cloud_ds'][()]
            new_cloud = cloudfile[task_index][step_index]['cloud'][()]
            new_cloud_ds = cloudfile[task_index][step_index]['cloud_ds'][()]

            fs = []
            reg_costs = []
            old_clouds_ds0 = []
            print "computing costs"
            for action in actionfile.keys():
                old_cloud = actionfile[action]['cloud_xyz']
                old_cloud_ds = clouds.downsample(old_cloud, DS_SIZE)
                f, corr = tps_registration.tps_rpm(old_cloud_ds[:,:3], new_cloud_ds[:,:3], n_iter=N_ITER_CHEAP, em_iter=EM_ITER_CHEAP, vis_cost_xy = tps_registration.ab_cost(old_cloud_ds, new_cloud_ds), user_data={'old_cloud':old_cloud_ds, 'new_cloud':new_cloud_ds})
                reg_cost = registration.tps_reg_cost(f)
                fs.append(f)
                reg_costs.append(reg_cost)
                old_clouds_ds0.append(old_cloud_ds)
            # sort based on reg_costs
            fs = [f for (cost, f) in sorted(zip(reg_costs, fs))]
            old_clouds_ds0 = [cloud for (cost, cloud) in sorted(zip(reg_costs, old_clouds_ds0))]
            
            print "plotting"
            x_nds = [cloud[:,:3] for cloud in old_clouds_ds0]
            y_md = new_cloud[:,:3]
            x_colors = [cloud[:,3:] for cloud in old_clouds_ds0]
            y_color = new_cloud[:,3:]
            plot_tps_registrations_proj_2d(x_nds, y_md, fs, (.1, .1, .04), x_colors, y_color)
 
    actionfile.close()
    cloudfile.close()   

if __name__ == "__main__":
    main()
