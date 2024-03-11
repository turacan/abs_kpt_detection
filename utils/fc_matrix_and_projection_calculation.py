import numpy as np

def calculate_projection_matrix(height=128, width=2048, fov_degrees=180):
    # Convert FOV to radians
    fov_radians = np.deg2rad(fov_degrees)
    # Calculate discretization and center coordinates
    # "-1" to ensure that the maximum pixel coordinates after transformation do not exceed the actual size of the image
    delta_phi = (width-1) / (2 * np.pi)
    delta_theta = (height-1) / (fov_radians)
    c_phi = (width-1) / 2   # shift to the right
    c_theta = (height-1) / 2    # shift downwards

    K = np.array([[delta_phi, 0, c_phi],
            [0, -delta_theta, c_theta],
            [0, 0, 1]])

    return K, fov_radians

def transform_kpts_to_unit_ray(kpts_uv: np.ndarray, proj_mat: np.ndarray):
    # Add an extra dimension to kpts_arr and K
    temp_kpt = np.column_stack((kpts_uv, np.ones(shape=(kpts_uv.shape[0],))))#.T
    
    K_inv = np.linalg.inv(proj_mat)

    phi_theta = np.matmul(K_inv, temp_kpt.T).T[:, :-1]
    phi = phi_theta[:, 0]
    theta = phi_theta[:, 1]

    ux = np.sin(np.pi/2-theta) * np.cos(phi) 
    uy = np.sin(np.pi/2-theta) * np.sin(phi)
    uz = np.cos(np.pi/2-theta)

    # Stack the coordinates to form the 3D points
    udvs = np.column_stack((ux, uy, uz))

    # Normalize the vectors to ensure they are unit vectors, range -1 to 1
    udvs /= np.linalg.norm(udvs, axis=1, keepdims=True)

    return udvs.astype(np.float32)