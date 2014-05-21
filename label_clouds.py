import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np
import argparse, h5py
import colorsys
import sys
import IPython as ipy

class Annotate(object):
    def __init__(self, cloud, selection_shape, z_color=False, unlabeled_color = (1,0,0), labeled_color = (0,0,1)):
        self.selection_shape = selection_shape
        self.unlabeled_color = np.array(unlabeled_color)
        self.labeled_color = np.array(labeled_color)
        assert not np.all(unlabeled_color == labeled_color)
        
        self.fig = plt.figure('label rope ends', figsize=(24,12))
        self.fig.clear()
        self.ax = self.fig.add_subplot(111, aspect='equal')
        
        if cloud.shape[1] == 6:
            self.cloud = cloud
        else:
            self.cloud = np.c_[cloud[:,:3], np.tile(self.unlabeled_color, (cloud.shape[0],1))]
        
        if z_color:
            z_min = self.cloud[:,2].min()
            z_max = self.cloud[:,2].max()
            self.z_color = np.empty((len(self.cloud),3))
            for i in range(len(self.cloud)):
                f = (self.cloud[i,2] - z_min) / (z_max - z_min)
                self.z_color[i,:] = np.array(colorsys.hsv_to_rgb(f,1,1))
        else:
            self.z_color = None
        self.scatters = []
        self.rescatter_cloud()
        self.ax.autoscale(False)
        
        if self.selection_shape == 'rectangle' or self.selection_shape == 'square':
            self.rect = Rectangle((0,0), 0, 0, facecolor='None', edgecolor='green', linestyle='dashed')
            self.ax.add_patch(self.rect)
        elif self.selection_shape == 'circle':
            self.circle = Circle((0,0), 0, facecolor='None', edgecolor='green', linestyle='dashed')
            self.ax.add_patch(self.circle)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None

        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.motion_notify_cid = None
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.show()
    
    def rescatter_cloud(self):
        for scatter in self.scatters:
            scatter.remove()
        self.scatters = []
        if self.z_color is not None:
            ul_inds = np.absolute((self.cloud[:,3:] - self.unlabeled_color)).sum(axis=1) == 0 # unlabeled inds
            self.scatters.append(plt.scatter(self.cloud[ul_inds,0], self.cloud[ul_inds,1], c=self.z_color[ul_inds,:], edgecolors=self.z_color[ul_inds,:], marker=',', s=5))
            self.scatters.append(plt.scatter(self.cloud[~ul_inds,0], self.cloud[~ul_inds,1], c=self.cloud[~ul_inds,3:], edgecolors=self.cloud[~ul_inds,3:], marker='o', s=20))
        else:
            self.scatters.append(plt.scatter(self.cloud[:,0], self.cloud[:,1], c=self.cloud[:,3:], edgecolors=self.cloud[:,3:], marker=',', s=5))
    
    def on_press(self, event):
        if event.xdata < self.ax.get_xlim()[0] or event.xdata > self.ax.get_xlim()[1] or \
            event.ydata < self.ax.get_ylim()[0] or event.ydata > self.ax.get_ylim()[1]:
            return
        self.x0 = event.xdata
        self.y0 = event.ydata
        self.motion_notify_cid = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_release(self, event):
        if self.motion_notify_cid:
            self.fig.canvas.mpl_disconnect(self.motion_notify_cid)
            self.motion_notify_cid = None
            if self.selection_shape == 'rectangle' or self.selection_shape == 'square':
                x0 = min(self.x0, self.x1)
                x1 = max(self.x0, self.x1)
                y0 = min(self.y0, self.y1)
                y1 = max(self.y0, self.y1)
                inside_rect_inds = (x0 <= self.cloud[:,0]) * (self.cloud[:,0] <= x1) * (y0 <= self.cloud[:,1]) * (self.cloud[:,1] <= y1)
                num_pts_inside_rect = inside_rect_inds.sum()
                self.cloud[inside_rect_inds,3:] = np.tile(self.labeled_color, (num_pts_inside_rect,1))
            elif self.selection_shape == 'circle':
                inside_circle_inds = np.apply_along_axis(np.linalg.norm, 1, self.cloud[:,:2] - np.array(self.circle.center)) <= self.circle.get_radius()
                num_pts_inside_circle = inside_circle_inds.sum()
                self.cloud[inside_circle_inds,3:] = np.tile(self.labeled_color, (num_pts_inside_circle,1))
            self.rescatter_cloud()
            plt.draw()
    
    def on_motion(self, event):
        if event.xdata < self.ax.get_xlim()[0] or event.xdata > self.ax.get_xlim()[1] or \
            event.ydata < self.ax.get_ylim()[0] or event.ydata > self.ax.get_ylim()[1]:
            return
        if self.selection_shape == 'rectangle':
            self.x1 = event.xdata
            self.y1 = event.ydata
        elif self.selection_shape == 'circle' or self.selection_shape == 'square':
            side = max(abs(event.xdata - self.x0), abs(event.ydata - self.y0))
            self.x1 = self.x0 + np.sign(event.xdata - self.x0) * side
            self.y1 = self.y0 + np.sign(event.ydata - self.y0) * side
        if self.selection_shape == 'rectangle' or self.selection_shape == 'square':
            self.rect.set_width(self.x1 - self.x0)
            self.rect.set_height(self.y1 - self.y0)
            self.rect.set_xy((self.x0, self.y0))
        elif self.selection_shape == 'circle':
            self.circle.center = ((self.x1 + self.x0)/2.0, (self.y0 + self.y1)/2.0)
            self.circle.set_radius(side/2.0)
        plt.draw()
    
    def on_key(self, event):
        if event.key == 'q':
            sys.exit(0)
        if event.key == 'd':
            plt.close(self.fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("--cloud_key", type=str, default="cloud_xyz")
    parser.add_argument('--selection_shape', type=str, choices=['rectangle', 'square', 'circle'], default='circle')
    parser.add_argument('--z_color', type=int, default=1, help="use hsv colors to visualize z-values for non-labeled points")

    args = parser.parse_args()
    
    file = h5py.File(args.file, 'a')
    for k in file:
        cloud = file[k][args.cloud_key][()]
        a = Annotate(cloud, args.selection_shape, args.z_color)
        del file[k][args.cloud_key]
        file[k][args.cloud_key] = a.cloud
    file.close()

if __name__ == "__main__":
    main()

