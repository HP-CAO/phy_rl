import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_trajectory(trajectory_tensor, reference_trajectory_tensor=None, name=''):
    """
   trajectory_tensor: an numpy array [n, 4], where n is the length of the trajectory,
                       5 is the dimension of each point on the trajectory, containing [x, x_dot, theta, theta_dot]
   """
    trajectory_tensor = np.array(trajectory_tensor)
    reference_trajectory_tensor = np.array(
        reference_trajectory_tensor) if reference_trajectory_tensor is not None else None
    n, c = trajectory_tensor.shape

    y_label_list = ["x", "x_dot", "theta", "theta_dot"]

    plt.figure(figsize=(6, 4))
    sns.set_style("darkgrid")

    plot, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
    n_points = 300

    ax1.plot(np.arange(n_points), trajectory_tensor[:n_points, 0], label="Phy-DRL controller")
    ax1.plot(np.arange(n_points), reference_trajectory_tensor[:n_points, 0], label="Model based controller")
    # ax.setylabel("X")
    ax1.set(ylabel="$x$")

    ax2.plot(np.arange(n_points), trajectory_tensor[:n_points, 1])
    ax2.plot(np.arange(n_points), reference_trajectory_tensor[:n_points, 1])
    # ax2.ylabel(r"$\dot{X}$")
    ax2.set(ylabel="$v$")

    ax3.plot(np.arange(n_points), trajectory_tensor[:n_points, 2])
    ax3.plot(np.arange(n_points), reference_trajectory_tensor[:n_points, 2])
    # ax3.ylabel(r"$\Theta$")
    ax3.set(ylabel="$\Theta$")

    ax4.plot(np.arange(n_points), trajectory_tensor[:n_points, 3])
    ax4.plot(np.arange(n_points), reference_trajectory_tensor[:n_points, 3])
    # ax4.ylabel(r"$\dot{\Theta}$")
    ax4.set(xlabel="Time steps $(k)$", ylabel="$w$")

    ax1.legend(loc="upper center", fontsize='medium',
               borderaxespad=0.,
               bbox_to_anchor=(0.5, 1.4),
               ncol=2,
               framealpha=0.0)

    plt.subplots_adjust(hspace=0.1)
    plt.grid(True)
    # plt.savefig(f"{name}_traj.pdf", format='pdf')
    plt.savefig(f"{name}_traj.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    res_trajectory = np.load("CDC/res.npy")
    linear_trajectory = np.load("CDC/linear.npy")
    plot_trajectory(res_trajectory, linear_trajectory, "compare")
