import seaborn as sns

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from matplotlib.legend_handler import HandlerBase
class MarkerHandler(HandlerBase):
        def create_artists(self, legend, orig_handle,
                    xdescent, ydescent, width, height, fontsize,
                    trans):
            return [plt.Line2D([width/2], [height/2.], linestyle='None',
                        marker=orig_handle[0], color=orig_handle[1], markersize=orig_handle[2])]


def plot_mpjpe_boxplot(err_arr: np.ndarray, IOU_THRESH: float=0.5):
    joint_list={0:'Pelvis', 
                1:'R_Hip', 2:'R_Knee', 3:'R_Ankle', 
                4:'L_Hip', 5:'L_Knee', 6:'L_Ankle', 
                7:'Torso', 8:'Neck', 9:'Head', 
                10:'R_Shoulder', 11:'R_Elbow', 12:'R_Wrist', 
                13:'L_Shoulder', 14:'L_Elbow', 15:'L_Wrist'
    }

    name=[]
    for k, v in joint_list.items():
        name.append(f'{k}: {v}')

    df_MPJPE_boxplot = pd.DataFrame(
        err_arr,
        columns=name
    )

    PJPE_mean = np.nanmean(err_arr, axis=0)
    df_MPJPE = pd.DataFrame({
        'Joint': list(range(16)),
        'Error': PJPE_mean
    })

    fig1 = plt.figure()
    fig1.set_size_inches(18.5, 10.5)
    ax = fig1.add_subplot()
    
    ax.set_title(f"Per Joint Position Error, Instance matching @IOU\u2265{IOU_THRESH}", fontsize=20)    # \u2265 greater equal sign
    ax.set_ylabel('Error [mm]', fontsize=16)
        # x label not used only x tcik labels -> will be set by boxlot automatically
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=12, rotation=45)    # x-axis label size and rotation applied to fit the long names better
    #plt.xticks(fontsize=13.5)

        # actual boxplot
    sns.boxplot(data=df_MPJPE_boxplot, fliersize=2, zorder=0) # has to be executed as first plot, otherwise it will occlude the other ones

    ax.set_ylim(0, 1000)
    # Enable grid with horizontal lines at each 0.1 interval
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.yticks([i for i in range(0, 1100, 100)])  # in 0 to 1m in 10 cm steps

    # plot with x-marker for each joint the mean position error
    ax.scatter(x=range(0,16), y=df_MPJPE['Error'], marker="x", linewidths=2, facecolor='red')

    MPJPE_skalar = np.nanmean(PJPE_mean)
    # plot overall MPJPE
    ax.axhline(y=MPJPE_skalar, linewidth=2, color='b', zorder=15)

    list_marker    = ["_", "x"]
    list_marker_color  = ["b", "r"]
    list_markersize = [12] * 2
    list_label    = ['MPJPE (mean error for all k joints)','PJPE (mean error for joint i)']

    ax.legend(list(zip(list_marker, list_marker_color, list_markersize)), list_label, fontsize=12,
            handler_map={tuple:MarkerHandler()}, loc="upper left") 

    ax.text(0.99, 0.99, f'MPJPE={MPJPE_skalar:.3f}mm',
        verticalalignment='top', horizontalalignment='right',
        transform=ax.transAxes, fontsize=16, color='b')

    plt.show()