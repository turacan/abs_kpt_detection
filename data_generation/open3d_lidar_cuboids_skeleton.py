#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Open3D Lidar visuialization example for CARLA"""

import glob
import os
import sys
import argparse
import time
from datetime import datetime
import random
import numpy as np
from matplotlib import cm
import open3d as o3d
import json
import logging

import queue

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

# 23 instances
LABEL_COLORS = np.array([
    (255, 255, 255), # None             # 0
    (70, 70, 70),    # Building         # 1
    (100, 40, 40),   # Fences           # 2
    (55, 90, 80),    # Other            # 3
    (220, 20, 60),   # Pedestrian       # 4
    (153, 153, 153), # Pole             # 5
    (157, 234, 50),  # RoadLines        # 6
    (128, 64, 128),  # Road             # 7
    (244, 35, 232),  # Sidewalk         # 8
    (107, 142, 35),  # Vegetation       # 9
    (0, 0, 142),     # Vehicle          # 10
    (102, 102, 156), # Wall             # 11
    (220, 220, 0),   # TrafficSign      # 12
    (70, 130, 180),  # Sky              # 13
    (81, 0, 81),     # Ground           # 14
    (150, 100, 100), # Bridge           # 15
    (230, 150, 140), # RailTrack        # 16
    (180, 165, 180), # GuardRail        # 17
    (250, 170, 30),  # TrafficLight     # 18
    (110, 190, 160), # Static           # 19
    (170, 120, 50),  # Dynamic          # 20
    (45, 60, 150),   # Water            # 21
    (145, 170, 100), # Terrain          # 22
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

def get_lidar_point(loc, w2s):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_sensor = np.dot(w2s, point)

    # New we must change from UE4's coordinate system
    #point_sensor[:, :1] = -point_sensor[:, :1]
    # and we remove the fourth componebonent also
    return point_sensor[0:3]

def get_all_static_actors(world):
    # get the blueprint library
    #blueprint_library = world.get_blueprint_library()
    labels = {}
    vehicle_actors = world.get_environment_objects(carla.CityObjectLabel.Vehicles)   # only intrested in static, 'parking' vehicles
    for npc in vehicle_actors:

        npc_id = npc.id         # unique id
        npc_type = npc.type     # carla.libcarla.CityObjectLabel.Vehicles
        npc_name = npc.name     # e.g. 'SM_Charger_parked_45_SM_0'
        bb = npc.bounding_box
        ex = bb.extent.x
        ey = bb.extent.y
        ez = bb.extent.z
        x = bb.location.x
        y = bb.location.y
        z = bb.location.z
        pitch = bb.rotation.pitch
        yaw = bb.rotation.yaw
        roll = bb.rotation.roll

        npc_dict = {}
        if "Harley" in npc_name:
            npc_dict["number_of_wheels"] = 2
        elif "Kawasaki" in npc_name:
            npc_dict["number_of_wheels"] = 2
        elif "Bike" in npc_name:
            npc_dict["number_of_wheels"] = 2
        elif "Vespa" in npc_name:
            npc_dict["number_of_wheels"] = 2
        elif "Yamaha" in npc_name:
            npc_dict["number_of_wheels"] = 2
        else:
            npc_dict["number_of_wheels"] = 4
        npc_dict["motion_state"] = "static"
        npc_dict["velocity"] = [0.0,0.0,0.0]
        npc_dict["acceleration"] = [0.0,0.0,0.0]
        npc_dict["extent"] = [ex,ey,ez]
        npc_dict["location"] = [x,y,z]
        npc_dict["rotation"] = [pitch,yaw,roll]
        npc_dict["semantic_tag"] = [10]     # carla.libcarla.CityObjectLabel.Vehicles   # npc_type
        npc_dict["type_id"] = npc_name
    
        #npc_dict["verts"] = [[v.x, v.y, v.z] for v in bb.get_world_vertices(npc.get_transform())]
        labels[npc_id] = npc_dict
    return labels

def save_lidar_and_labels(point_cloud, world, frame_id, meta, save_path, town):
    save_path = save_path + os.path.sep + town
    # get the blueprint library
    blueprint_library = world.get_blueprint_library()
    # make save path dir
    labels_path = os.path.join(save_path, "labels")
    calib_path = os.path.join(save_path, "calib")   # world2sensor transformation matrix
    pcl_path = os.path.join(save_path, "semantic_point_cloud")  # ply 
    raw_path = os.path.join(save_path, "raw")   # raw pcl 
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)
    os.makedirs(calib_path, exist_ok=True)
    os.makedirs(pcl_path, exist_ok=True)
    os.makedirs(raw_path, exist_ok=True)
    
    '''carla.SemanticLidarMeasurement Inherited from carla.SensorData
    point_cloud.raw_data: raw_data (bytes)
    Received list of raw detection points. Each point consists of [x,y,z] coordinates plus the cosine of the incident angle, 
        the index of the hit actor, and its semantic tag.'''
    data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))       #data[:]['ObjIdx']
    np.save(os.path.join(raw_path,'%.6d.npy' % frame_id),data)

    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Open3D uses a right-handed coordinate system
    #points = np.array([data['x'], -data['y'], data['z']]).T
    points = np.array([data['x'], data['y'], data['z']]).T
    # # An example of adding some noise to our data if needed:
    # points += np.random.uniform(-0.05, 0.05, size=points.shape)

    # Colorize the pointcloud based on the CityScapes color palette
    labels = np.array(data['ObjTag'])
    int_color = LABEL_COLORS[labels]

    # test = [elem for elem in labels if (isinstance(elem, list) or isinstance(elem, np.ndarray))]
    # # In case you want to make the color intensity depending
    # # of the incident ray angle, you can use:
    # int_color *= np.array(data['CosAngle'])[:, None]
    point_list = o3d.geometry.PointCloud()
    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)

    #if point_cloud.frame % 1 == 0:
    w2s = np.array(point_cloud.transform.get_matrix())  # 4x4-transformation matrix from local to global coordinates
    meta["world2sensor"] = w2s.tolist()
    o3d.io.write_point_cloud(os.path.join(pcl_path,'%.6d.ply' % frame_id), point_list)
    json.dump(meta, open(os.path.join(calib_path,'%.6d.json' % frame_id), 'w' ) )
    # first get all actors and assume them as static
    labels = get_all_static_actors(world)
    # get all dynamic walkeras and vehicles as actors
    vehicle_actors = world.get_actors().filter('*vehicle.*')
    
    walker_actors = world.get_actors().filter('*walker.*')
    actors = list(vehicle_actors) + list(walker_actors)
    
    # add dynamic actors, overwrite existing actors with static
    count_error_elems = 0
    for npc in actors:
        transform = npc.get_transform()
        rotation = transform.rotation
        location = transform.location           
        npc_id = npc.id

        if not npc_id in data[:]['ObjIdx']:
            count_error_elems +=1

        # npc_semantic_tag: list of which semantic objects the instance exists, 
            # can be empty if e.g. obj == walker.ai
            # len(npc_semantic_tag) == 1, e.g. car, walker
            # len(npc_semantic_tag) > 1, traffic.traffic_light, consists of TrafficLight, TrafficSign, Pole
        npc_semantic_tag = npc.semantic_tags    

        npc_type_id = npc.type_id   # vehicle model name, e.g. vehicle.tesla.model3
        npc_velocity = npc.get_velocity()
        npc_acceleration = npc.get_acceleration()
        bb = npc.bounding_box
        
        ex = bb.extent.x
        ey = bb.extent.y
        ez = bb.extent.z
        x = location.x + bb.location.x
        y = location.y + bb.location.y
        z = location.z + bb.location.z
        pitch = rotation.pitch + bb.rotation.pitch
        yaw = rotation.yaw + bb.rotation.yaw
        roll = rotation.roll + bb.rotation.roll

        vx = npc_velocity.x
        vy = npc_velocity.y
        vz = npc_velocity.z
        ax = npc_acceleration.x
        ay = npc_acceleration.y
        az = npc_acceleration.z
        
        npc_dict = {}
        if isinstance(npc, carla.Walker):
            bones = npc.get_bones()
            # prepare the bones (get name and world position)
            boneIndex = {}  
            for i, bone in enumerate(bones.bone_transforms):
                boneIndex[bone.name] = {"world": [bone.world.location.x, bone.world.location.y, bone.world.location.z]}
            npc_dict["bones"] = boneIndex   # exclusive for walker
        if isinstance(npc, carla.Vehicle):
            bp = blueprint_library.find(npc_type_id)
            npc_dict["number_of_wheels"] = int(bp.get_attribute('number_of_wheels'))
        npc_dict["motion_state"] = "dynamic"
        npc_dict["velocity"] = [vx,vy,vz]
        npc_dict["acceleration"] = [ax,ay,az]
        npc_dict["extent"] = [ex,ey,ez]
        npc_dict["location"] = [x,y,z]
        npc_dict["rotation"] = [pitch,yaw,roll]
        npc_dict["semantic_tag"] = npc_semantic_tag
        npc_dict["type_id"] = npc_type_id
        
        #npc_dict["verts"] = [[v.x, v.y, v.z] for v in bb.get_world_vertices(npc.get_transform())]
        labels[npc_id] = npc_dict

    json.dump(labels, open(os.path.join(labels_path,'%.6d.json' % frame_id), 'w' ) )
    #print(count_error_elems)
    return count_error_elems

def puppeteer(world, blending):
    for ped in world.get_actors().filter('*walker.*'):

        # make some transition from custom pose to animation
        ped.blend_pose(np.sin(blending))

        
    
    

def generate_lidar_bp(arg, world, blueprint_library, delta):
    """Generates a CARLA blueprint based on the script parameters"""
    if arg.semantic:
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
    else:
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        if arg.no_noise:
            lidar_bp.set_attribute('dropoff_general_rate', '0.0')
            lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
            lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
        else:
            lidar_bp.set_attribute('noise_stddev', '0.2')

    lidar_bp.set_attribute('upper_fov', str(arg.upper_fov))
    lidar_bp.set_attribute('lower_fov', str(arg.lower_fov))
    lidar_bp.set_attribute('channels', str(arg.channels))
    lidar_bp.set_attribute('range', str(arg.range))
    lidar_bp.set_attribute('rotation_frequency', str(1.0 / delta))
    lidar_bp.set_attribute('points_per_second', str(arg.points_per_second*arg.channels*arg.frame_rate))
    #lidar_bp.set_attribute('sensor_tick', str(0.1))
    return lidar_bp


def add_open3d_axis(vis):
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)


def main(arg):
    args = arg
    """Main function of the script"""
    client = carla.Client(arg.host, arg.port)
    client.set_timeout(20.0)
    synchronous_master = False

    if isinstance(args.town, str):
        args.town = [args.town]
    elif args.town == None:
        temp = client.get_available_maps()
        args.town = [elem.split(os.path.sep)[-1] for elem in temp if "Opt" in elem]

    arg.save_path = arg.save_path + os.path.sep + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(client.get_available_maps())
    for town in args.town:
        #world = client.get_world()
        try:
            world = client.load_world(town)
    
            lidar_queue = queue.Queue()
            vehicles_list = []
            walkers_list = []
            all_id = []
            original_settings = world.get_settings()
            settings = world.get_settings()
            traffic_manager = client.get_trafficmanager(8000)
            traffic_manager.set_synchronous_mode(True)

            delta = 1.0/arg.frame_rate

            settings.fixed_delta_seconds = delta
            settings.synchronous_mode = True
            settings.no_rendering_mode = arg.no_rendering
            world.apply_settings(settings)


            # -------------
            # Spawn Recording Vehicle
            # -------------
            
            blueprint_library = world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter(arg.filter)[0]
            vehicle_transform = random.choice(world.get_map().get_spawn_points())
            vehicle = world.spawn_actor(vehicle_bp, vehicle_transform)
            vehicle.set_autopilot(True)

            # lidar_bp = generate_lidar_bp(arg, world, blueprint_library, delta)

            # user_offset = carla.Location(arg.x, arg.y, arg.z)
            # vehicle_offset = carla.Location(x=-0.5, z=1.8)
            # mounting_offset = vehicle_offset + user_offset
            # rotation = carla.Rotation(pitch=arg.pitch, roll=args.roll, yaw=args.yaw)
            # lidar_transform = carla.Transform(vehicle_offset + user_offset, rotation)

            # lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

            # lidar.listen(lidar_queue.put)


            #######
            blueprints = get_actor_blueprints(world, args.filterv, args.generationv)
            blueprintsWalkers = get_actor_blueprints(world, args.filterw, args.generationw)

            if args.safe:
                blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
                blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
                blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
                blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
                blueprints = [x for x in blueprints if not x.id.endswith('t2')]
                blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
                blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
                blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]

            blueprints = sorted(blueprints, key=lambda bp: bp.id)

            spawn_points = world.get_map().get_spawn_points()   # only spawn points for vehicles
            number_of_spawn_points = len(spawn_points)

            if args.number_of_vehicles < number_of_spawn_points:
                random.shuffle(spawn_points)
            elif args.number_of_vehicles > number_of_spawn_points:
                msg = 'requested %d vehicles, but could only find %d spawn points'
                logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
                args.number_of_vehicles = number_of_spawn_points

            # @todo cannot import these directly.
            SpawnActor = carla.command.SpawnActor
            SetAutopilot = carla.command.SetAutopilot
            FutureActor = carla.command.FutureActor

            # --------------
            # Spawn vehicles
            # --------------
            batch = []
            hero = args.hero
            for n, transform in enumerate(spawn_points):
                if n >= args.number_of_vehicles:
                    break
                blueprint = random.choice(blueprints)
                if blueprint.has_attribute('color'):
                    color = random.choice(blueprint.get_attribute('color').recommended_values)
                    blueprint.set_attribute('color', color)
                if blueprint.has_attribute('driver_id'):
                    driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                    blueprint.set_attribute('driver_id', driver_id)
                if hero:
                    blueprint.set_attribute('role_name', 'hero')
                    hero = False
                else:
                    blueprint.set_attribute('role_name', 'autopilot')

                # spawn the cars and set their autopilot and light state all together
                batch.append(SpawnActor(blueprint, transform)
                    .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

            for response in client.apply_batch_sync(batch, synchronous_master):
                if response.error:
                    logging.error(response.error)
                else:
                    vehicles_list.append(response.actor_id)

            # Set automatic vehicle lights update if specified
            if args.car_lights_on:
                all_vehicle_actors = world.get_actors(vehicles_list)
                for actor in all_vehicle_actors:
                    traffic_manager.update_vehicle_lights(actor, True)

            # -------------
            # Spawn Walkers
            # -------------
            # some settings
            percentagePedestriansRunning = 0.05      # how many pedestrians will run
            percentagePedestriansCrossing = 0.3     # how many pedestrians will walk through the road
            if args.seedw:
                world.set_pedestrians_seed(args.seedw)
                random.seed(args.seedw)
            else:
                world.set_pedestrians_seed(random.randint(0, 5))
                random.seed(args.seedw)
            # 1. take all the random locations to spawn
            spawn_points = []
            spawn_points_valid_check = []
            for i in range(args.number_of_walkers):
                spawn_point = carla.Transform()
                loc = world.get_random_location_from_navigation()
                temp = (round(loc.x, 0), round(loc.y, 0))
                while (temp in spawn_points_valid_check) or loc == None:
                    loc = world.get_random_location_from_navigation()
                    temp = (round(loc.x, 0), round(loc.y, 0))

                spawn_point.location = loc
                spawn_points.append(spawn_point)
                spawn_points_valid_check.append(temp)


            # 2. we spawn the walker object
            batch = []
            walker_speed = []
            for spawn_point in spawn_points:
                walker_bp = random.choice(blueprintsWalkers)
                # set as not invincible
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                # set the max speed
                if walker_bp.has_attribute('speed'):
                    if (random.random() > percentagePedestriansRunning):
                        # walking
                        walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                    else:
                        # running
                        walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
                else:
                    print("Walker has no speed")
                    walker_speed.append(0.0)
                batch.append(SpawnActor(walker_bp, spawn_point))
            results = client.apply_batch_sync(batch, True)
            walker_speed2 = []
            for i in range(len(results)):
                if results[i].error:
                    logging.error(results[i].error) # if error with obj, don't spawn the actor and don't add to walker list
                else:
                    walkers_list.append({"id": results[i].actor_id})
                    walker_speed2.append(walker_speed[i])
            walker_speed = walker_speed2
            # 3. we spawn the walker controller
            batch = []
            walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
            for i in range(len(walkers_list)):
                batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
            results = client.apply_batch_sync(batch, True)
            for i in range(len(results)):
                if results[i].error:
                    logging.error(results[i].error)
                else:
                    walkers_list[i]["con"] = results[i].actor_id
            # 4. we put together the walkers and controllers id to get the objects from their id
            for i in range(len(walkers_list)):
                all_id.append(walkers_list[i]["con"])
                all_id.append(walkers_list[i]["id"])
            all_actors = world.get_actors(all_id)

            # wait for a tick to ensure client receives the last transform of the walkers we have just created
            # if args.asynch or not synchronous_master:
            #     world.wait_for_tick()
            # else:
            #world.tick()

            # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
            # set how many pedestrians can cross the road
            world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
            for i in range(0, len(all_id), 2):
                # start walker
                all_actors[i].start()
                # set walk to random point
                all_actors[i].go_to_location(world.get_random_location_from_navigation())
                # max speed
                all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

            print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

            # Example of how to use Traffic Manager parameters
            traffic_manager.global_percentage_speed_difference(30.0)

            # # -------------
            # # Spawn Recording Vehicle
            # # -------------
            
            # blueprint_library = world.get_blueprint_library()
            # vehicle_bp = blueprint_library.filter(arg.filter)[0]
            # vehicle_transform = random.choice(world.get_map().get_spawn_points())
            # vehicle = world.spawn_actor(vehicle_bp, vehicle_transform)
            # vehicle.set_autopilot(True)

            lidar_bp = generate_lidar_bp(arg, world, blueprint_library, delta)

            user_offset = carla.Location(arg.x, arg.y, arg.z)
            vehicle_offset = carla.Location(x=-0.5, z=1.8)
            mounting_offset = vehicle_offset + user_offset
            rotation = carla.Rotation(pitch=arg.pitch, roll=args.roll, yaw=args.yaw)
            lidar_transform = carla.Transform(vehicle_offset + user_offset, rotation)

            lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

            lidar.listen(lidar_queue.put)

            meta = {}
            meta["vehicle"] = str(arg.filter)
            meta["frame_rate"] = arg.frame_rate
            meta["mounting_offset"] = np.array([mounting_offset.x, mounting_offset.y, mounting_offset.z]).tolist()
            meta["mounting_angle"] = np.array([rotation.pitch, rotation.roll, rotation.yaw]).tolist()

            frame = 0
            dt0 = datetime.now()
            blending = 0
            
            while frame < arg.n_frames:
                #puppeteer(world, blending)
                frame_id = world.tick()
                
                point_cloud = lidar_queue.get()
                count_error_elems = 0
                #point_cloud = lidar_queue.get()
                if frame_id % args.save_nth_frame == 0:
                    # move the pedestrian
                    blending += 0.015
                    count_error_elems = save_lidar_and_labels(point_cloud,world,frame_id, meta, arg.save_path, town)
                    frame += 1
                    
                process_time = datetime.now() - dt0
                sys.stdout.write('\r' + 'FPS: ' + str(1.0 / process_time.total_seconds()) + ' World Frame: ' + str(frame_id)+ ' Sensor Frame: '+ str(point_cloud.frame_number) + ' count_error_elems:' + str(count_error_elems))
                sys.stdout.flush()
                dt0 = datetime.now()
                #frame += 1     

            #print("destroyer", town)
            # world.apply_settings(original_settings)
            # traffic_manager.set_synchronous_mode(False)
            # for actor in all_actors:
            #     actor.destroy()

            # vehicle.destroy()
            # lidar.destroy()

        except Exception as ex:
            print(town, ex)
        finally:
            print("destroyer", town)
            if not args.asynch and synchronous_master:
                settings = world.get_settings()
                settings.synchronous_mode = False
                settings.no_rendering_mode = False
                settings.fixed_delta_seconds = None
                world.apply_settings(settings)

            print('\ndestroying %d vehicles' % len(vehicles_list))
            client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

            # stop walker controllers (list is [controller, actor, controller, actor ...])
            for i in range(0, len(all_id), 2):
                all_actors[i].stop()

            print('\ndestroying %d walkers' % len(walkers_list))
            client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

            vehicle.destroy()
            lidar.destroy()

            # print("destroyer", town)
            # world.apply_settings(original_settings)
            # traffic_manager.set_synchronous_mode(False)

            # vehicle.destroy()
            # lidar.destroy()
            #vis.destroy_window()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port of CARLA Simulator (default: 2000)')
    argparser.add_argument(
        '--no-rendering',
        action='store_true',
        help='use the no-rendering mode which will provide some extra'
        ' performance but you will lose the articulated objects in the'
        ' lidar, such as pedestrians')
    argparser.add_argument(
        '--semantic',
        default=True,
        action='store_true',
        help='use the semantic lidar instead, which provides ground truth'
        ' information')
    argparser.add_argument(
        '--no-noise',
        action='store_true',
        help='remove the drop off and noise from the normal (non-semantic) lidar')
    argparser.add_argument(
        '--no-autopilot',
        action='store_false',
        help='disables the autopilot so the vehicle will remain stopped')
    argparser.add_argument(
        '--show-axis',
        action='store_true',
        help='show the cartesian coordinates axis')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='model3',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--upper-fov',
        default=90.0, #90.0
        type=float,
        help='lidar\'s upper field of view in degrees (default: 15.0)')
    argparser.add_argument(
        '--lower-fov',
        default=-90.0, #-90.0
        type=float,
        help='lidar\'s lower field of view in degrees (default: -25.0)')
    argparser.add_argument(
        '--channels',
        default=1024.0,
        type=float,
        help='lidar\'s channel count (default: 64)')
    argparser.add_argument(
        '--range',
        default=250.0,
        type=float,
        help='lidar\'s maximum range in meters (default: 100.0)')
    argparser.add_argument(
        '--points-per-second',
        default=2024+128,   # 2024+128
        type=int,
        help='lidar\'s points per second (default: 500000)')
    argparser.add_argument(
        '--n_frames',
        default=150,    # 100
        type=int,
        help='number of frames to save (default: 100)')
    argparser.add_argument(
        '--save_nth_frame',
        default=40, # 25
        type=int,
        help='frames to be saved (default: 10')
    argparser.add_argument(
        '-x',
        default=0.0,
        type=float,
        help='offset in the sensor position in the X-axis in meters (default: 0.0)')
    argparser.add_argument(
        '-y',
        default=0.0,
        type=float,
        help='offset in the sensor position in the Y-axis in meters (default: 0.0)')
    argparser.add_argument(
        '-z',
        default=0.0,
        type=float,
        help='offset in the sensor position in the Z-axis in meters (default: 0.0)')
    argparser.add_argument(
        '-pitch',
        default=0.0,
        type=float,
        help='pitch of the sensor in degrees (default: 0.0)')
    argparser.add_argument(
        '-roll',
        default=0.0,
        type=float,
        help='roll of the sensor in degrees (default: 0.0)')
    argparser.add_argument(
        '-yaw',
        default=0.0,
        type=float,
        help='yaw of the sensor in degrees (default: 0.0)')
    argparser.add_argument(
        '-frame_rate',
        default=10.0,
        type=float,
        help='frame rate')
    argparser.add_argument(
        '-save_path',
        default="/media/nvme/carla/dataset/CARLA_HIGH_RES_LIDAR/",
        type=str,
        help='save path')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=40, # 30
        type=int,
        help='Number of vehicles (default: 30)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=300, # 50
        type=int,
        help='Number of walkers (default: 10)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='Avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='Filter vehicle model (default: "vehicle.*")')
    argparser.add_argument(
        '--generationv',
        metavar='G',
        default='All',
        help='restrict to certain vehicle generation (values: "1","2","All" - default: "All")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='Filter pedestrian type (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--generationw',
        metavar='G',
        default='2',
        help='restrict to certain pedestrian generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='Port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--asynch',
        action='store_true',
        help='Activate asynchronous mode execution')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Activate hybrid mode for Traffic Manager')
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        type=int,
        help='Set random device seed and deterministic mode for Traffic Manager')
    argparser.add_argument(
        '--seedw',
        metavar='S',
        default=None,  # 0
        type=int,
        help='Set the seed for pedestrians module')
    argparser.add_argument(
        '--pedestrians_cross_factor',
        metavar='S',
        default=0.05,
        type=float,
        help='Sets the percentage of pedestrians that can walk on the road or cross at any point on the road. Value should be between 0.0 and 1.0. For example, a value of 0.1 would allow 10% of pedestrians to walk on the road. Default is 0.0')
    argparser.add_argument(
        '--town',
        metavar='S',
        default=['Town10HD_Opt'], #"Town10HD_Opt",   ['Town01_Opt', Town02_Opt', 'Town03_Opt', 'Town04_Opt', 'Town05_Opt', 'Town10HD_Opt']
        type=str,
        help='Set the town')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=False,
        help='Enable automatic car light management')
    argparser.add_argument(
        '--hero',
        action='store_true',
        default=False,
        help='Set one of the vehicles as hero')
    argparser.add_argument(
        '--respawn',
        action='store_true',
        default=False,
        help='Automatically respawn dormant vehicles (only in large maps)')

    args = argparser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')