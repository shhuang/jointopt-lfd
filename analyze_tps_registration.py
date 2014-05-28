import matplotlib.pyplot as plt
import numpy as np
import argparse, h5py
import eval_util
from rapprentice import registration, tps_registration, clouds, plotting_plt
import IPython as ipy

N_ITER_CHEAP = 14 # do_task_eval uses 8 # TODO
EM_ITER_CHEAP = 1

def plot_tps_registrations_proj_2d(x_clouds, y_clouds, fs, res, texts):
    # set interactive
    plt.ion()
    
    k = len(x_clouds) # number of demonstrations
    
    fig, axes = plt.subplots(num='2d projection plot', facecolor='white', figsize=(18, 18), nrows=k, ncols=4)
    if len(axes.shape) == 1:
        axes= axes[None,:]

    for i, (x_cloud, y_cloud, f, text) in enumerate(zip(x_clouds, y_clouds, fs, texts)):
        x_nd = x_cloud[:,:3]
        x_color = x_cloud[:,3:]
        y_md = y_cloud[:,:3]
        y_color = y_cloud[:,3:]

        axes[i,0].clear()
        axes[i,0].axis('off')
        axes[i,0].set_aspect('equal')
        axes[i,0].scatter(x_nd[:,0], x_nd[:,1], c=x_color, edgecolors='none', marker=',', s=5)
    
        grid_means = .5 * (x_nd.max(axis=0) + x_nd.min(axis=0))
        grid_mins = grid_means - (x_nd.max(axis=0) - x_nd.min(axis=0))
        grid_maxs = grid_means + (x_nd.max(axis=0) - x_nd.min(axis=0))
        x_median = np.median(x_nd, axis=0)
        plt.sca(axes[i,0])
        plotting_plt.plot_warped_grid_proj_2d(lambda xyz: xyz, grid_mins[:2], grid_maxs[:2], z=x_median[2], xres=res[0], yres=res[1], draw=False)
    
        axes[i,1].clear()
        axes[i,1].axis('off')
        axes[i,1].set_aspect('equal')
        axes[i,1].scatter(y_md[:,0], y_md[:,1], c=y_color, edgecolors='none', marker=',', s=5)
    
        axes[i,2].clear()
        axes[i,2].set_xlim(axes[i,1].get_xlim())
        axes[i,2].set_ylim(axes[i,1].get_ylim())
        axes[i,2].axis('off')
        axes[i,2].set_aspect('equal')
        xwarped_nd = f.transform_points(x_nd)
        axes[i,2].scatter(xwarped_nd[:,0], xwarped_nd[:,1], c=x_color, edgecolors='none', marker=',', s=5)
        plt.sca(axes[i,2])
        plotting_plt.plot_warped_grid_proj_2d(f.transform_points, grid_mins[:2], grid_maxs[:2], z=x_median[2], xres=res[0], yres=res[1], draw=False)

        axes[i,3].clear()
        axes[i,3].axis('off')
        axes[i,3].text(0.5, 0.5, '%s\n cost %.4f\n reg_cost %.4f'%(text, f._cost, registration.tps_reg_cost(f)),
            horizontalalignment='center',
            verticalalignment='center')

    fig.tight_layout()
    
    plt.draw()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('actionfile', type=str)
    parser.add_argument('cloudfile', type=str, help="result file containing the encountered clouds")
    parser.add_argument('type', type=str, choices=['explore', 'compare'], help="")
    parser.add_argument("--tasks", type=int, nargs='*', metavar="i_task")
    parser.add_argument("--taskfile", type=str)
    parser.add_argument("--i_start", type=int, default=-1, metavar="i_task")
    parser.add_argument("--i_end", type=int, default=-1, metavar="i_task")
    parser.add_argument("--steps", type=int, nargs='*', metavar="i_step")
    parser.add_argument("--draw_rows", type=int, default=8, help="plot only the draw_rows registrations with the smallest cost")
    parser.add_argument("--ds_size", type=float, default=0.025, metavar="meters")

    args = parser.parse_args()
    
    actionfile = h5py.File(args.actionfile, 'r')
    cloudfile = h5py.File(args.cloudfile, 'r')
    cloud_items = eval_util.get_holdout_items(cloudfile, args.tasks, args.taskfile, args.i_start, args.i_end)
    
    if args.type == 'explore':
        fs = []
        reg_costs = []
        old_clouds_ds = []
        new_clouds_ds = []
        texts = []
        for i_task, _ in cloud_items:
            n_steps = len(cloudfile[i_task]) - (1 if 'init' in cloudfile[i_task] else 0)
            steps = [i_step for i_step in args.steps if i_step in range(n_steps)] if args.steps else range(n_steps)
            for i_step in steps:
                if args.steps and i_step not in args.steps:
                    continue
                print "task %s step %i" % (i_task, i_step)
    
                old_cloud = cloudfile[i_task][str(i_step)]['demo_cloud'][()]
                old_cloud_ds = clouds.downsample(old_cloud, args.ds_size)
                new_cloud = cloudfile[i_task][str(i_step)]['cloud'][()]
                new_cloud_ds = clouds.downsample(new_cloud, args.ds_size)
    
                f, corr = tps_registration.tps_rpm(old_cloud_ds[:,:3], new_cloud_ds[:,:3], n_iter=N_ITER_CHEAP, em_iter=EM_ITER_CHEAP, vis_cost_xy = tps_registration.ab_cost(old_cloud_ds, new_cloud_ds), user_data={'old_cloud':old_cloud_ds, 'new_cloud':new_cloud_ds})
                reg_cost = registration.tps_reg_cost(f)
                fs.append(f)
                reg_costs.append(reg_cost)
                old_clouds_ds.append(old_cloud_ds)
                new_clouds_ds.append(new_cloud_ds)
                texts.append("task %s step %i" % (i_task, i_step))
    
                if args.draw_rows == len(fs) or (i_task == cloud_items[-1][0] and i_step == steps[-1]):
                    # sort based on reg_costs
                    fs = [f for (cost, f) in sorted(zip(reg_costs, fs))]
                    old_clouds_ds = [cloud for (cost, cloud) in sorted(zip(reg_costs, old_clouds_ds))]
                    new_clouds_ds = [cloud for (cost, cloud) in sorted(zip(reg_costs, new_clouds_ds))]
                    texts = [text for (cost, text) in sorted(zip(reg_costs, texts))]
        
                    print "plotting"
                    plot_tps_registrations_proj_2d(old_clouds_ds, new_clouds_ds, fs, (.1, .1, .04), texts)
                    ipy.embed()
                    
                    fs[:] = []
                    reg_costs[:] = []
                    old_clouds_ds[:] = []
                    new_clouds_ds[:] = []
                    texts[:] = []
    elif args.type == 'compare':
        for i_task, _ in cloud_items:
            for i_step in range(len(cloudfile[i_task]) - (1 if 'init' in cloudfile[i_task] else 0)):
                if args.steps and i_step not in args.steps:
                    continue
                print "task %s step %i" % (i_task, i_step)
     
                new_cloud = cloudfile[i_task][str(i_step)]['cloud'][()]
                new_cloud_ds = clouds.downsample(new_cloud, args.ds_size)
                
                fs = []
                reg_costs = []
                old_clouds_ds = []
                new_clouds_ds = []
                texts = []
                print "computing costs"
                for action in actionfile.keys():
                    old_cloud = actionfile[action]['cloud_xyz']
                    old_cloud_ds = clouds.downsample(old_cloud, args.ds_size)
                    f, corr = tps_registration.tps_rpm(old_cloud_ds[:,:3], new_cloud_ds[:,:3], n_iter=N_ITER_CHEAP, em_iter=EM_ITER_CHEAP, vis_cost_xy = tps_registration.ab_cost(old_cloud_ds, new_cloud_ds), user_data={'old_cloud':old_cloud_ds, 'new_cloud':new_cloud_ds})
                    reg_cost = registration.tps_reg_cost(f)
                    fs.append(f)
                    reg_costs.append(reg_cost)
                    old_clouds_ds.append(old_cloud_ds)
                    new_clouds_ds.append(new_cloud_ds)
                    texts.append("task %s step %i action %s" % (i_task, i_step, action))
                # sort based on reg_costs
                fs = [f for (cost, f) in sorted(zip(reg_costs, fs))]
                old_clouds_ds = [cloud for (cost, cloud) in sorted(zip(reg_costs, old_clouds_ds))]
                if args.draw_rows != -1:
                    fs = fs[:args.draw_rows]
                    old_clouds_ds = old_clouds_ds[:args.draw_rows]
                
                print "plotting"
                plot_tps_registrations_proj_2d(old_clouds_ds, new_clouds_ds, fs, (.1, .1, .04), texts)
                ipy.embed()
 
    actionfile.close()
    cloudfile.close()   

if __name__ == "__main__":
    main()
