import numpy as np

BONE_KEYS = [
        'crl_hips__C',        # Pelvis              # 0
        'crl_thigh__R',       # R_Hip               # 1
        'crl_leg__R',         # R_Knee              # 2
        'crl_foot__R',        # R_Ankle             # 3
        'crl_thigh__L',       # L_Hip               # 4
        'crl_leg__L',         # L_Knee              # 5
        'crl_foot__L',        # L_Ankle             # 6
        'crl_spine01__C',     # Torso               # 7
        'crl_neck__C',        # Neck                # 8
        'crl_Head__C',        # Head                # 9
        'crl_arm__R',         # R_Shoulder          # 10
        'crl_foreArm__R',     # R_Elbow             # 11
        'crl_hand__R',        # R_Wrist             # 12
        'crl_arm__L',         # L_Shoulder          # 13
        'crl_foreArm__L',     # L_Elbow             # 14
        'crl_hand__L'         # L_Wrist             # 15
]

# bone/ joints connections
KINTREE_TABLE = np.array([ 
    [0, 1],     # Pelvis -> R_Hip
    [1, 2],     # R_Hip -> R_Knee
    [2, 3],     # R_Knee -> R_Ankle
    [0, 4],     # Pelvis -> L_Hip
    [4, 5],     # L_Hip -> L_Knee
    [5, 6],     # L_Knee -> L_Ankle
    [0, 7],     # Pelvis -> Torso
    [7, 8],     # Torso -> Neck
    [8, 9],     # Neck -> Head
    [7, 10],    # Torso -> R_Shoulder
    [10, 11],   # R_Shoulder -> R_Elbow
    [11, 12],   # R_Elbow -> R_Wrist
    [7, 13],    # Torso -> L_Shoulder
    [13, 14],   # L_Shoulder -> L_Elbow
    [14, 15]    # L_Elbow -> L_Wrist
]).T

KEYPOINT_NAMES = [
    'Pelvis',
    'R_Hip', 
    'R_Knee', 
    'R_Ankle',
    'L_Hip',
    'L_Knee',
    'L_Ankle',
    'Torso',
    'Neck',
    'Head',
    'R_Shoulder',
    'R_Elbow',
    'R_Wrist',
    'L_Shoulder',
    'L_Elbow',
    'L_Wrist'
]

KEYPOINT_CONNECTION_RULES = [
    ('Pelvis', 'R_Hip', (255, 0, 0)),
    ('R_Hip', 'R_Knee', (255, 0, 0)),
    ('R_Knee', 'R_Ankle', (255, 0, 0)),

    ('Pelvis', 'L_Hip', (0, 0, 255)),
    ('L_Hip', 'L_Knee', (0, 0, 255)),
    ('L_Knee', 'L_Ankle', (0, 0, 255)),

    ('Pelvis', 'Torso', (0, 255, 0)),
    ('Torso', 'Neck', (0, 255, 0)),
    ('Neck', 'Head', (0, 255, 0)),

    ('Torso', 'R_Shoulder', (255, 0, 0)),
    ('R_Shoulder', 'R_Elbow', (255, 0, 0)),
    ('R_Elbow', 'R_Wrist', (255, 0, 0)),

    ('Torso', 'L_Shoulder', (0, 0, 255)),
    ('L_Shoulder', 'L_Elbow', (0, 0, 255)),
    ('L_Elbow', 'L_Wrist', (0, 0, 255))
]

KEYPOINT_FLIP_MAP  = [
    #('Pelvis', 'Pelvis'),
    ('R_Hip', 'L_Hip'),
    ('R_Knee', 'L_Knee'),
    ('R_Ankle', 'L_Ankle'),
    ('Torso', 'Torso'),
    #('Neck', 'Neck'),
    #('Head', 'Head'),
    ('R_Shoulder', 'L_Shoulder'),
    ('R_Elbow', 'L_Elbow'),
    ('R_Wrist', 'L_Wrist')
]
