                               # get bounding box extent of maskedArr 
                # area = pycocotools.mask.area(rle)
                # [x, y, w, h] = cv2.boundingRect(maskedArr)

                # key = str(key)
                # # Find contours in the binary mask
                # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #CHAIN_APPROX_SIMPLE, CHAIN_APPROX_NONE
                # # Convert contours to polygon representation
                # segmentation = []
                # for contour in contours:    
                #     contour = contour.reshape(-1, 2)

                #     area = cv2.contourArea(contour)
                #     # Check if the contour has at least 6 points (3 coordinates) and a valid size to form a valid polygon
                #     if area < 4 or contour.shape[0] < 6:  
                #         continue

                #     contour = contour.flatten().tolist()
                #     segmentation.append(contour)    # list[list[float]]
                # if len(segmentation) == 0: 
                #     continue
                
                # visu contours
                # if visu_flag:
                # test = np.uint8(255*hha_img)
                # test = cv2.applyColorMap(test, cv2.COLORMAP_BONE)
                # for polygon in segmentation:
                #     polygon = np.array(polygon, dtype=int)
                #     polygon = polygon.reshape(-1, 2)
                #     for x,y in polygon:
                #         # Draw the polygon on the image using a blue line with thickness 5
                        
                #         test[y,x] = [255, 0, 0] #cv2.polylines(test, [var], isClosed=True, color=(255, 0, 0), thickness=5)
                # test = cv2.resize(test, (1500, 750))
                # mask = cv2.resize((mask*255).astype(np.uint8), (1500, 750))

                # cv2.imshow("Image with polygon", test)
                # cv2.imshow("mask", mask)
                # cv2.waitKey(0)
                    
                # segment_mask = pycocotools.mask.encode(np.asarray(mask, order="F"))
                # # Get the bounding box for the current connected component
                # box = pycocotools.mask.toBbox(segment_mask)    # Format: box[:2] -> Left-top corner, box[2:] -> width, height
                # bbox_xywh = (box[0], box[1], box[2], box[3]) # (x, y, width, height)
                # # Convert the bounding box to XYXY_ABS format   # Left Top, Right Down
                # # left_top = (box[0], box[1])

                # # Convert to BoxMode.XYXY_ABS
                # xmin, ymin, xmax, ymax = bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2]-1, bbox_xywh[1] + bbox_xywh[3]-2
                # offset = 0
                # box = [xmin-offset, ymin-offset, xmax+offset, ymax+offset]


















 # get difference between cuboid center location and pelvis joint
                # if differnce is too huge, raw data sensor data and world state metadata are not synchronized
                # TODO: check whether to skip this frame or make an adjustment in joint location using cuboid centroid location
                # diff_u = np.maximum(crl_hips__C_u, uv_cuboid_center[0]) - np.minimum(crl_hips__C_u, uv_cuboid_center[0])
                # diff_v = np.maximum(crl_hips__C_v, uv_cuboid_center[1]) - np.minimum(crl_hips__C_v, uv_cuboid_center[1])
                # if diff_u > hha_img.shape[1]/3: # object extent probably go over the image edges, thus both locations are so seperated
                #     diff_u = hha_img.shape[1]-1 - np.maximum(crl_hips__C_u, uv_cuboid_center[0])
                #     diff_u += np.minimum(crl_hips__C_u, uv_cuboid_center[0])
                
                # if diff_v > hha_img.shape[0]/3: # typically not applicable, check if boundaries goes over the image edges
                #     diff_v = hha_img.shape[0]-1 - np.maximum(crl_hips__C_v, uv_cuboid_center[1])
                #     diff_v += np.minimum(crl_hips__C_v, uv_cuboid_center[1])
                
                # if crl_hips__C_u < uv_cuboid_center[0]: 
                #     diff_u *= -1
                # if crl_hips__C_v < uv_cuboid_center[1]: 
                #     diff_v *= -1
                
                # TESTING: skip frame if differnce is too huge, threshold = 10px
                # if box_area<20: # small obj
                #     if (abs(diff_u)/box_width > 0.2) or (abs(diff_v)/box_height > 0.2): # higher threshold with smaller instances, bc error will be larger out of box
                #         counter_misplaced +=1
                # else:
                #     if ((abs(diff_u)/box_width > 0.1) or (abs(diff_v)/box_height > 0.1)):
                #         print(box_area, abs(diff_u)/box_width, abs(diff_v)/box_height)
                #         counter_misplaced +=1
                #     #continue

                # obj idx tag image extents of raw data not the same as joint locations from metadata
                # along u-coord     






                # uv_set = set()
                # keypoints_list = []
                # black_img = np.full(shape=(hha_img.shape[:2]), fill_value=0, dtype=np.uint8)
                # for (u_float, v_float) in uv_skeleton:
                #     if (abs(diff_u) > 4) or (abs(diff_v) > 4):  # only if diff is noticeable, test wise if diff larger than 4 pixels
                #         u_float = u_float - diff_u    # additionally for joint corrections  
                #         v_float = v_float - diff_v
                #         if u_float > hha_img.shape[1]-1:
                #             u_float = u_float - (hha_img.shape[1]-1)
                #         if u_float < 0:
                #             u_float = hha_img.shape[1] + u_float
                #             if u_float > hha_img.shape[1]-1:    # e.g u_float was -0.4
                #                 u_float = 0
                #                 print("if u_float > hha_img.shape[1]-1")    # observe if there is any error
                #         if v_float > hha_img.shape[0]-1:
                #             v_float = v_float - (hha_img.shape[0]-1)
                #         if v_float < 0:
                #             v_float = hha_img.shape[0] + v_float
                #             if v_float > hha_img.shape[0]-1:
                #                 v_float = 0
                #     if (abs(diff_u) > 50) or (abs(diff_v) > 50):
                #         print('big diff between bbox centre and measured joints')
                #     u, v = int(u_float), int(v_float) 
                    
                #     black_img[v, u] = 255
                #     keypoints_list.extend([float(u_float), float(v_float)]) # needs to be JSON serializable 

                #     # label joint visibility 
                #     # v=0: not labeled (in which case x=y=0), v=1: labeled but not visible, and v=2: labeled and visible.
                #     if (u, v) not in uv_set:
                #         uv_set.add((u, v))                    
                #         if xyz_things[v, u, -2] == 4:
                #             keypoints_list.extend([2])
                #             final_img[v, u]  = [255, 0, 255]    # visu
                #         else: # occluded by some oject
                #             keypoints_list.extend([1])
                #             final_img[v, u]  = [255, 100, 100]  # visu
                #     else:
                #         keypoints_list.extend([1])  
                
                # # for comparison
                # visible_jnts['2'] += np.argwhere(np.array(keypoints_list[2::3])==2).size
                # if np.argwhere(np.array(keypoints_list[2::3])==1).size  > 5:

                #     visible_jnts['1'] += np.argwhere(np.array(keypoints_list[2::3])==1).size 

                # final_img[uv_cuboid_center_int[1], uv_cuboid_center_int[0]]  = [0, 0, 255]  # colorize pelvis center of obj
                # obj['keypoints'] = keypoints_list
                # # if not (np.argwhere(np.array(keypoints_list[2::3]) == 2).size >= 2):
                # #     print()

                # # adjust 
                # x_axis = np.sort(np.unique(np.array(keypoints_list[::3], dtype=int)))
                # diff_x_axis = np.diff(x_axis)
                # split_idx = np.argwhere(diff_x_axis>int(hha_img.shape[0]/2)).squeeze()
                # if split_idx.size > 0:
                #     split_1  = x_axis[:split_idx+1]
                #     split_2 = x_axis[split_idx+1:]

                #     x1 = split_2[0]
                #     x2 = split_1[-1]

                # else:
                #     x1 = x_axis[0]
                #     x2 = x_axis[-1]

                # # along y-axis
                # y_axis = np.sort(np.unique(np.array(keypoints_list[1::3], dtype=int)))
                # y1 = y_axis[0]
                # y2 = y_axis[-1]
                
                # #box = [x1, y1, x2, y2]
                # box = np.round(np.array([np.minimum(box[0], x1), # left
                #     np.minimum(box[1], y1), # top
                #     np.maximum(box[2], x2), # right
                #     np.maximum(box[3], y2)  # bottom
                #     ])).astype(int)
                
                # offset = 1,  # one free pixel between obj extrema and bbox
                # box = [int(box[0]-offset), int(box[1]-offset), int(box[2]+offset), int(box[3]+offset)]

                # boxes[f'{int(key)}'] = box  # list idx 







                # min_keypoints_u = min(keypoints_list[::3])True  # left
                # min_keypoints_v = min(keypoints_list[1::3]) # top

                # max_keypoints_u = max(keypoints_list[::3])  # right
                # max_keypoints_v = max(keypoints_list[1::3]) # bottom

                # box = np.round(np.array([np.minimum(box[0], min_keypoints_u), # left
                #     np.minimum(box[1], min_keypoints_v), # top
                #     np.maximum(box[2], max_keypoints_u), # right
                #     np.maximum(box[3], max_keypoints_v)  # bottom
                #     ])).astype(int)
                
                # offset = 2  # one free pixel between obj extrema and bbox
                # box = [int(box[0]-offset), int(box[1]-offset), int(box[2]+offset), int(box[3]+offset)]
                
                # bbox go over whole image, make two instances out of one 
                # if split_idx.size > 0:
                #     box = [int(x1), int(y1), int(hha_img.shape[1]-1), int(y2)]
                #     obj['bbox'] = box
                #     annotations.append(obj) 

                #     if visu_flag:
                #         test = cv2.rectangle(test, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 255, 255], 1)
                #         final_img = cv2.rectangle(final_img, (int(obj['bbox'][0]), int(obj['bbox'][1])), (int(obj['bbox'][2]), int(obj['bbox'][3])), [255, 255, 255], 1)

                #     box = [0, int(y1), int(x2), int(y2)]
                #     obj['bbox'] = box
                #     annotations.append(obj) 
                #     if visu_flag:
                #         test = cv2.rectangle(test, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 255, 255], 1)
                #         final_img = cv2.rectangle(final_img, (int(obj['bbox'][0]), int(obj['bbox'][1])), (int(obj['bbox'][2]), int(obj['bbox'][3])), [255, 255, 255], 1)
                # else:
                #     obj['bbox'] = box
                #     annotations.append(obj) 
            
                #     if visu_flag:
                #         test = cv2.rectangle(test, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 255, 255], 1)
                #         final_img = cv2.rectangle(final_img, (int(obj['bbox'][0]), int(obj['bbox'][1])), (int(obj['bbox'][2]), int(obj['bbox'][3])), [255, 255, 255], 1)
                #         # cv2.imshow("Bounding Rectangle", final_img)
                #         # cv2.waitKey(0)
                # black_img[uv_cuboid_center_int[1], uv_cuboid_center_int[0]] = 100      
                # black_img = cv2.rectangle(black_img, (int(obj['bbox'][0]), int(obj['bbox'][1])), (int(obj['bbox'][2]), int(obj['bbox'][3])), 255, 1)
                # if visu_flag:
                #     keypoints_list_arr = np.array(keypoints_list).reshape(-1, 3)[:, :-1].astype(int)
                #     binary_mask = pycocotools.mask.decode(rle)
                #     binary_mask = (np.stack([binary_mask, binary_mask, binary_mask], axis=-1) * 255).astype(np.uint8)
                #     binary_mask[keypoints_list_arr[:, 1], keypoints_list_arr[:, 0]] = [255, 0, 255]
                #     binary_mask[uv_cuboid_center_int[1], uv_cuboid_center_int[0]] = [0, 255, 0]
                #     # Naming a window 
                #     cv2.namedWindow("binary_mask", cv2.WINDOW_NORMAL) 
                #     cv2.resizeWindow("binary_mask", 1920, 1080) 
                #     cv2.imshow('binary_mask', binary_mask)
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()

                #     # cv2.imshow("Image skelleton in u,v-coordinates", black_img)
                #     # cv2.waitKey(0)
                #     # cv2.destroyAllWindows()

                # area_box = (box[2] - box[0]) * (box[3] - box[1])
                # if area_box > 1e6:#:*hha_img.shape[1]:
                #     print('whoa thats a huge bbox')
                # # calculate offset between joints from recording and adjust to be centered at obj center
                # #v_center_obj, u_center_obj