import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from readgri import readgri
import argparse, sys


def plotmesh(mesh: dict, args: dict):
    V = mesh['V']; E = mesh['E']; BE = mesh['BE']
    fig, ax = plt.subplots()
    plt.triplot(V[:,0], V[:,1], E, 'k-')
    for i in range(BE.shape[0]):
        plt.plot(V[BE[i,0:2],0], V[BE[i,0:2],1], '-', linewidth=1, color='black')
    if args['metric']:
        plot_metric_fields(mesh, ax)
    plt.axis('equal')
    plt.tick_params(axis='both', labelsize=12)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_metric_fields(mesh: dict, ax: plt.Axes):
    V = mesh['V']; E = mesh['E']
    for i, element in enumerate(E):
        verts = [V[v_ind] for v_ind in element]  # (x,y) coordinate for each vertex
        centroid = np.asarray([np.mean([v[0] for v in verts]), np.mean([v[1] for v in verts])])
        delta_x = [verts[i] - verts[i-1] for i in range(len(verts))]
        # Assume all delta x are actually a unit length L=1 apart, solve for M
        A_mat = [[vec[0]**2, 2*vec[0]*vec[1], vec[1]**2] for vec in delta_x]
        A_mat = np.reshape(np.asarray(A_mat), (3,3))
        soln_abc = np.matmul(np.linalg.inv(A_mat), np.ones((3,1))).flatten()
        M_mat = np.array([[soln_abc[0], soln_abc[1]], [soln_abc[1], soln_abc[2]]])
        eig_vals, eig_vecs = np.linalg.eig(M_mat)
        major_ax_i = np.argmax(eig_vals)
        minor_ax_i = int(not major_ax_i)
        axes_scales = np.zeros(len(eig_vals))
        axes_scales[major_ax_i] = eig_vals[major_ax_i]**(-1/2)
        axes_scales[minor_ax_i] = eig_vals[minor_ax_i]**(-1/2)
        # axes_scales[major_ax_i] = max([np.linalg.norm(x) for x in delta_x])
        # axes_scales[minor_ax_i] = axes_scales[major_ax_i] * (eig_vals[minor_ax_i]/eig_vals[major_ax_i])
        angle_major = np.arctan2(eig_vecs[major_ax_i][1], eig_vecs[major_ax_i][0])
        # Only plot every 10th ellipse as to not overcrowd the plot
        if i % 10 == 0:
            plt.scatter(centroid[0], centroid[1], color='red', alpha=0.5, s=2)
            ellipse = Ellipse(centroid, axes_scales[major_ax_i], axes_scales[minor_ax_i],
                              angle=-angle_major*180/np.pi, facecolor='blue', alpha=0.5)
            ax.add_patch(ellipse)


def get_euclidean_dist(a: np.ndarray, b: np.ndarray):
    return np.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)


if __name__ == "__main__":
    # Ingest command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='Name of the .gri file to be plotted', required=True)
    parser.add_argument('-m', '--metric', help='Plot the metric field', required=False, action='store_true')
    args = vars(parser.parse_args())
    # Plot the mesh file
    plt.style.use('fast')
    mesh = readgri(args['file'])
    plotmesh(mesh, args)
