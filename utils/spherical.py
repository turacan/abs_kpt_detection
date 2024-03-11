import numpy as np
import open3d as o3d

def o3d_draw_skeleton(joints3D, kintree_table, color_set="RGB"):

    colors = []
    if color_set == "CMY":
        left_right_mid = [[1.0,1.0,0.0], [0.0,1.0,1.0], [1.0,0.0,1.0]]
    else:
        left_right_mid = [[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]]
    kintree_colors = [0,0,0,2,2,2,1,1,1,0,0,0,2,2,2]#kintree_table.shape[1]*[0]# # [0,1,2,0,1,2,0,1,2,1,2,0,0,1,2,1,2,1,2,1,2]
    for c in kintree_colors:
        colors.append(left_right_mid[c])
        
    
    # For each joint
    lines = []
    points = []
    for i in range(0, kintree_table.shape[1]):
        j1 = kintree_table[0][i]
        j2 = kintree_table[1][i]
        lines.append([j1, j2])
    points = joints3D[:, :]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    joints = o3d.geometry.PointCloud()
    joints.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    joints.colors = o3d.utility.Vector3dVector(colors)
    return line_set, joints


def to_deflection_coordinates(x,y,z):
    # To cylindrical
    p = np.sqrt(x ** 2 + y ** 2)
    # To spherical 
    phi = np.arctan2(y, x)   
    theta = -np.arctan2(p, z) + np.pi/2
    return phi, theta
    
def estimate_n_bins(arr, arr_min, arr_max ):
    # https://academic.oup.com/biomet/article-abstract/66/3/605/232642
    R = arr_max - arr_min
    n = arr.size
    std = np.std(arr)
    
    return int(R*(n**(1/2))/(3.49*std))

def get_xyz(pc):
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]
    return x,y,z

def spherical_projection(pc, height=128, width=2048, fov_degrees=None, th=0.1, sort_largest_first=False, bins_h=None, max_range=None):
    '''spherical projection 
    Args:
        pc: point cloud, dim: N*C
    Returns:
        pj_img: projected spherical iamges, shape: h*w*C
    '''

    fov_radians = np.deg2rad(fov_degrees)
    #theta_range = [-1.6, 1.6]
    theta_range = [round(-fov_radians / 2,1), round(fov_radians / 2,1)]

    # filter all small range values to avoid overflows in theta min max calculation
    #if isinstance(theta_range, type(None)):
        
    r = np.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2 + pc[:, 2] ** 2)
    arr1inds = r.argsort()
    if sort_largest_first:
        pc = pc[arr1inds]
    else:
        pc = pc[arr1inds[::-1]]
    #pc = pc[arr1inds]
    r = np.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2 + pc[:, 2] ** 2)
    if not isinstance(max_range,type(None)):
        indices = np.where((r > th)*(r<=max_range))
    else:
        indices = np.where(r > th)
    pc = pc[indices]
        
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        
    phi, theta = to_deflection_coordinates(x,y,z)

    #indices = np.where(r > th)
    if isinstance(theta_range, type(None)):
        theta_min, theta_max = [theta.min(), theta.max()]
    else: 
        theta_min, theta_max = theta_range
        
    #phi_min, phi_max = [phi.min(), phi.max()]
    phi_min, phi_max = [-np.pi, np.pi]
    
    # assuming uniform distribution of rays
    if isinstance(bins_h, type(None)):
        bins_h = np.linspace(theta_min, theta_max, height)[::-1]
    bins_w = np.linspace(phi_min, phi_max, width)[::-1]
    
    theta_img = np.stack(width*[bins_h], axis=-1)
    phi_img = np.stack(height*[bins_w], axis=0)

    idx_h = np.digitize(theta, bins_h)-1
    idx_w = np.digitize(phi, bins_w)-1
    
    pj_img = np.zeros((height, width, pc.shape[1])).astype(np.float32)
    
    pj_img[idx_h, idx_w, :] = pc
    pj_img[idx_h, idx_w, :] = pc

   
    alpha = np.sqrt(np.square(theta_img)+np.square(phi_img))
   
    return pj_img#, alpha, (theta_min, theta_max), (phi_min, phi_max) 


def project_points_to_seg_image(pc, sem_seg, height, width, th=0.001,
                                theta_min_max=[-2.0,-1.5], phi_min_max=[-np.pi, np.pi]):
    '''spherical back projection of points to labels
    Args:
        pc: point cloud, dim: [N_points, 3]
        sem_seg: semantic image, dim: [height, width, channel]
    Returns:
        labels: projected spherical iamges, shape: [N_points, channel]
    '''

    #height, width = sem_seg.shape
    
    N = pc.shape[0]
    labels = np.zeros((N, 1), dtype=np.uint32)
    
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]
    phi, theta = to_deflection_coordinates(x,y,z)
    
    theta_min, theta_max = theta_min_max
    phi_min, phi_max = phi_min_max
    
    # assuming uniform distribution of rays
    bins_h = np.linspace(theta_min, theta_max, height)[::-1]
    bins_w = np.linspace(phi_min, phi_max, width)[::-1]

    idx_h = np.digitize(theta, bins_h)-1
    idx_w = np.digitize(phi, bins_w)-1

    for i in range(idx_h.shape[0]):
        try:
            labels[i] = sem_seg[idx_h[i], idx_w[i]]
        except:
            continue
    return labels#, idx_h, idx_w

def get_indices(pc, height, width, th=0.001, theta_min_max=[-2.0,-1.5], phi_min_max=[-np.pi, np.pi]):
    '''spherical back projection of points to labels
    Args:
        pc: point cloud, dim: [N_points, 3]
        sem_seg: semantic image, dim: [height, width, channel]
    Returns:
        labels: projected spherical iamges, shape: [N_points, channel]
    '''

    #height, width = sem_seg.shape
    
    N = pc.shape[0]
    
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]
    phi, theta = to_deflection_coordinates(x,y,z)
    
    theta_min, theta_max = theta_min_max
    phi_min, phi_max = phi_min_max
    
    # assuming uniform distribution of rays
    bins_h = np.linspace(theta_min, theta_max, height)[::-1]
    bins_w = np.linspace(phi_min, phi_max, width)[::-1]

    idx_h = np.digitize(theta, bins_h)-1
    idx_w = np.digitize(phi, bins_w)-1
    
    return idx_w, idx_h