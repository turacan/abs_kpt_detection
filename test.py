import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

# Assuming you have a SphericalProjectionImage with x, y, and z channels
# For the sake of this example, let's generate a random image
image = cv2.imread("test_scene.exr", cv2.IMREAD_UNCHANGED)
image = image[450:670, 689:1530]
# image = np.random.rand(256, 256, 3)

x = image[:, :, 0]
y = image[:, :, 1]
z = image[:, :, 2]

# 1. Gradient Computation
Sxx = cv2.Scharr(x, cv2.CV_64F, 1, 0)
Sxy = cv2.Scharr(x, cv2.CV_64F, 0, 1)
Syx = cv2.Scharr(y, cv2.CV_64F, 1, 0)
Syy = cv2.Scharr(y, cv2.CV_64F, 0, 1)
Szx = cv2.Scharr(z, cv2.CV_64F, 1, 0)
Szy = cv2.Scharr(z, cv2.CV_64F, 0, 1)

# 2. Cross Product for Surface Normal
normal_x = Syx * Szy - Szx * Syy
normal_y = Szx * Sxy - Szy * Sxx
normal_z = Sxx * Syy - Syx * Sxy

# 3. Normalization
n = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
normal_x /= n
normal_y /= n
normal_z /= n

# 4. Computing the Direction Cosine
cos_theta = normal_z

# Visualization
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

axs[0].imshow(image)
axs[0].set_title('SphericalProjectionImage')

axs[1].imshow(np.sqrt(Sxx**2 + Sxy**2), cmap='gray')
axs[1].set_title('Gradient Magnitude of x')

axs[2].imshow(np.sqrt(normal_x**2 + normal_y**2 + normal_z**2), cmap='gray')
axs[2].set_title('Surface Normal Magnitude')

axs[3].imshow(cos_theta, cmap='gray')
axs[3].set_title('Direction Cosine')

for ax in axs:
    ax.axis('off')

plt.tight_layout()
plt.show()



# import os
# import json
# import glob
# from tqdm import tqdm
# root = '/workspace/data/dataset'

# for splittype in ['train', 'test']:
#     files = glob.glob(os.path.join(root, splittype, 'labels/*')) 
    
#     for file in tqdm(files):
#         with open(file, 'r+') as f:
#             data = json.load(f)

#             file_name = data['file_name'] 
#             real_save_path = file_name.split(os.path.sep)
#             del real_save_path[-4]
#             real_save_path = (os.path.sep).join(real_save_path)

#             data['file_name'] = real_save_path
#             # Move the file pointer to the beginning of the file for writing
#             f.seek(0)

#             # Write the updated dictionary back to the JSON file
#             json.dump(data, f)

#             # Truncate the remaining content in case the new data is shorter than the previous content
#             f.truncate()