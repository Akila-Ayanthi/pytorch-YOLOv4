import cv2
import numpy as np
from xml.dom import minidom
import os
import subprocess
import matplotlib.pyplot as plt

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Object(object):
    def __init__(self, obj_id):
        self.obj_id = obj_id
        self.times = []
        self.cam_1_frame_ids = []
        self.cam_2_frame_ids = []
        self.cam_3_frame_ids = []
        self.cam_4_frame_ids = []
        self.cam_1_coord = []
        self.cam_2_coord = []
        self.cam_3_coord = []
        self.cam_4_coord = []

    def add_points(self, x, y, t, cam_1_frame_id, cam_2_frame_id, cam_3_frame_id, cam_4_frame_id):
        self.times.append(t)
        self.cam_1_frame_ids.append(cam_1_frame_id)
        self.cam_2_frame_ids.append(cam_2_frame_id)
        self.cam_3_frame_ids.append(cam_3_frame_id)
        self.cam_4_frame_ids.append(cam_4_frame_id)
        

        cmd = '/home/dissana8/LAB/project1 /home/dissana8/LAB/LAB_Calibration/LAB_calib/c1.calib ' + str(x) + ' ' + str(y) + ' 0.0'
        result = subprocess.check_output(cmd, shell=True)
        s = result[8:-1]
        s=s.decode("utf-8")
        image_x = float(s.split(',')[0])
        image_y = float(s.split(',')[1])
        self.cam_1_coord.append(Point(image_x, image_y))

        cmd = '/home/dissana8/LAB/project1 /home/dissana8/LAB/LAB_Calibration/LAB_calib/c2.calib ' + str(x) + ' ' + str(y) + ' 0.0'
        result = subprocess.check_output(cmd, shell=True)
        s = result[8:-1]
        s=s.decode("utf-8")
        image_x = float(s.split(',')[0])
        image_y = float(s.split(',')[1])
        self.cam_2_coord.append(Point(image_x, image_y))

        cmd = '/home/dissana8/LAB/project1 /home/dissana8/LAB/LAB_Calibration/LAB_calib/c3.calib ' + str(x) + ' ' + str(y) + ' 0.0'
        result = subprocess.check_output(cmd, shell=True)
        s = result[8:-1]
        s=s.decode("utf-8")
        image_x = float(s.split(',')[0])
        image_y = float(s.split(',')[1])
        self.cam_3_coord.append(Point(image_x, image_y))

        cmd = '/home/dissana8/LAB/project1 /home/dissana8/LAB/LAB_Calibration/LAB_calib/c4.calib ' + str(x) + ' ' + str(y) + ' 0.0'
        result = subprocess.check_output(cmd, shell=True)
        s = result[8:-1]
        s=s.decode("utf-8")
        image_x = float(s.split(',')[0])
        image_y = float(s.split(',')[1])
        self.cam_4_coord.append(Point(image_x, image_y))


    def get_id(self):
        return self.obj_id

    def print_traj(self):
        print('\t', self.get_id(), ',cam_1_coord: ', len(self.cam_1_coord), ',cam_2_coord: ', len(self.cam_2_coord),
              ',cam_3_coord: ', len(self.cam_3_coord), ',cam_4_coord: ', len(self.cam_4_coord))

    def get_coords(self):
        c1_x = []
        c1_y = []
        c2_x = []
        c2_y = []
        c3_x = []
        c3_y = []
        c4_x = []
        c4_y = []

        for point in self.cam_1_coord:
            c1_x.append(point.x)
            c1_y.append(point.y)

        for point in self.cam_2_coord:
            c2_x.append(point.x)
            c2_y.append(point.y)

        for point in self.cam_3_coord:
            c3_x.append(point.x)
            c3_y.append(point.y)

        for point in self.cam_4_coord:
            c4_x.append(point.x)
            c4_y.append(point.y)

        return (c1_x, c1_y, c2_x, c2_y, c3_x, c3_y, c4_x, c4_y, self.times, self.cam_1_frame_ids, self.cam_2_frame_ids,
                self.cam_3_frame_ids, self.cam_4_frame_ids)

def check_if_object_exists(obj_id, obj_list):
    idx = -1
    found = False
    for obj in obj_list:
        idx += 1
        crnt_id = obj.get_id()
        if obj_id == crnt_id:
            found = True
            break
    if found:
        return idx
    else:
        return -1


def add_detection(obj_list, obj_id, x, y, t, cam_1_frame_id, cam_2_frame_id, cam_3_frame_id, cam_4_frame_id):
    idx = check_if_object_exists(obj_id, obj_list)

    if idx == -1:
        # object doesn't exist in the list
        obj = Object(obj_id)
        obj.add_points(x, y, t, cam_1_frame_id, cam_2_frame_id, cam_3_frame_id, cam_4_frame_id)
        obj_list.append(obj)

    else:
        obj_list[idx].add_points(x, y, t, cam_1_frame_id, cam_2_frame_id, cam_3_frame_id, cam_4_frame_id)

def save_coords(object_list, exp_name):
    # file_id = 0
    # first_run=True
    # for o in object_list:
    #     print('saving objects '+str(file_id)+ '/'+str(len(object_list))+'...')
    #     obj_id = o.get_id()
    #     cam_1_coord = o.cam_1_coord
    #     cam_2_coord = o.cam_2_coord
    #     cam_3_coord = o.cam_3_coord
    #     cam_4_coord = o.cam_4_coord
        
    #     cam_1_x, cam_1_y, cam_2_x, cam_2_y, cam_3_x, cam_3_y, cam_4_x, cam_4_y,\
    #         times, cam_1_frame_ids, cam_2_frame_ids, cam_3_frame_ids, cam_4_frame_ids = o.get_coords()        
        
    #     file_id +=1

    save_path = "/home/dissana8/LAB/data2/"+exp_name
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    xc_cam1=[]
    yc_cam1=[]
    xc_cam2=[]
    yc_cam2=[]
    xc_cam3=[]
    yc_cam3=[]
    xc_cam4=[]
    yc_cam4=[]
    cam_1_frame_id=[]
    cam_2_frame_id=[]
    cam_3_frame_id=[]
    cam_4_frame_id=[]

    for obj in object_list:
        counter = -1
        obj_id = obj.obj_id
        cam_1_l = obj.cam_1_coord
        cam_2_l = obj.cam_2_coord
        cam_3_l = obj.cam_3_coord
        cam_4_l = obj.cam_4_coord
        print('saving coordinates of : ' + str(obj_id) + ' .....')
        for i in range(len(cam_1_l)):
            counter += 1
            print('\t frame',counter,' ....')
            cam_1_p = cam_1_l[i]
            cam_2_p = cam_2_l[i]
            cam_3_p = cam_3_l[i]
            cam_4_p = cam_4_l[i]
            xc_cam1.append(cam_1_p.x)
            yc_cam1.append(cam_1_p.y)
            xc_cam2.append(cam_2_p.x)
            yc_cam2.append(cam_2_p.y)
            xc_cam3.append(cam_3_p.x)
            yc_cam3.append(cam_3_p.y)
            xc_cam4.append(cam_4_p.x)
            yc_cam4.append(cam_4_p.y)

            cam_1_frame_id.append(obj.cam_1_frame_ids[i])
            cam_2_frame_id.append(obj.cam_2_frame_ids[i])
            cam_3_frame_id.append(obj.cam_3_frame_ids[i])
            cam_4_frame_id.append(obj.cam_4_frame_ids[i])

    np.save(save_path+'/cam1_coords', np.c_[cam_1_frame_id, xc_cam1, yc_cam1])
    np.save(save_path+'/cam2_coords', np.c_[cam_2_frame_id, xc_cam2, yc_cam2])
    np.save(save_path+'/cam3_coords', np.c_[cam_3_frame_id, xc_cam3, yc_cam3])
    np.save(save_path+'/cam4_coords', np.c_[cam_4_frame_id, xc_cam4, yc_cam4])


def print_obj_list(obj_list):
    print('---------- \t -------------')
    for obj in obj_list:
        obj.print_traj()
    print('---------- \t -------------')


def findClosest(time, camera_time_list):
    val = min(camera_time_list, key=lambda x: abs(x - time))
    return camera_time_list.index(val)

def extract_tracks_from_Annotations(path,file_name):
    obj_list = []

    #===== process the index files of camera 1 ======#
    with open('/home/dissana8/LAB/Visor/cam1/index.dmp') as f:
        content = f.readlines()
    cam_content = [x.strip() for x in content]
    c1_frames = []
    c1_times = []
    for line in cam_content:
#         print(line)
        s = line.split(" ")
        frame = s[0]
        time = float(s[1]+'.'+s[2])
        c1_frames.append(frame)
        c1_times.append(time)
        #print("once")
        #break
    #print("two")    

    with open('/home/dissana8/LAB/Visor/cam2/index.dmp') as f:
        content = f.readlines()

    cam_content = [x.strip() for x in content]
    c2_frames = []
    c2_times = []
    for line in cam_content:
        s = line.split(" ")
        frame = s[0]
        time = float(s[1]+'.'+s[2])
        c2_frames.append(frame)
        c2_times.append(time)
        #print("once")
        #break
    #print("two")

    # ===== process the index files of camera 3 ======#
    with open('/home/dissana8/LAB/Visor/cam3/index.dmp') as f:
        content = f.readlines()

    cam_content = [x.strip() for x in content]
    c3_frames = []
    c3_times = []
    for line in cam_content:
        s = line.split(" ")
        frame = s[0]
        time = float(s[1] + '.' + s[2])
        c3_frames.append(frame)
        c3_times.append(time)
        #print("once")
        #break
    #print("two")

    # ===== process the index files of camera 4 ======#
    with open('/home/dissana8/LAB/Visor/cam4/index.dmp') as f:
        content = f.readlines()

    cam_content = [x.strip() for x in content]
    c4_frames = []
    c4_times = []
    for line in cam_content:
        s = line.split(" ")
        frame = s[0]
        time = float(s[1] + '.' + s[2])
        c4_frames.append(frame)
        c4_times.append(time)
        #print("once")
        #break
    #print("two")
    #===== process the GT annotations  =======#
    with open("/home/dissana8/LAB/"+file_name) as f:
        content = f.readlines()
        

    content = [x.strip() for x in content]
    counter = -1
    print('Extracting GT annotation ...')
    for line in content:
        counter += 1
        print(str(counter) +'/'+ str(len(content)))
        print(s)
        s = line.split(" ")
        
        time = float(s[0])
        frame_idx = findClosest(time, c1_times) # we have to map the time to frame number
        c1_frame_no = c1_frames[frame_idx]
        
        frame_idx = findClosest(time, c2_times)  # we have to map the time to frame number
        c2_frame_no = c2_frames[frame_idx]
        

        frame_idx = findClosest(time, c3_times)  # we have to map the time to frame number
        c3_frame_no = c3_frames[frame_idx]

        
        frame_idx = findClosest(time, c4_times)  # we have to map the time to frame number
        c4_frame_no = c4_frames[frame_idx]
        

        s = s[1:] # get the rest of the string i.e <obj_id>, <x_world>, <y_world>, <z_world>
        for i in range(0, len(s), 4):
            obj_id = s[i]
            print(obj_id)
            #=== removing this bit as the names are given as text names i.e Oswald ==#
            obj_id = int(obj_id[2:]) # trim out the ID part in text. i.e 'ID00228'
            x = float(s[i+1])/1000
            y = float(s[i + 2]) / 1000

            add_detection(obj_list, obj_id, x, y, time, c1_frame_no, c2_frame_no, c3_frame_no, c4_frame_no)
    

    return obj_list    

def visualise_GT_tracks(path, path_2, path_3, path_4, obj_list):

    for obj in obj_list:
        counter = -1
        obj_id = obj.obj_id
        cam_1_l = obj.cam_1_coord
        cam_2_l = obj.cam_2_coord
        cam_3_l = obj.cam_3_coord
        cam_4_l = obj.cam_4_coord
        print('plotting the trajectory of : ' + str(obj_id) + ' .....')
        for i in range(len(cam_1_l)):
            counter += 1
            print('\t frame',counter,' ....')
            cam_1_p = cam_1_l[i]
            cam_2_p = cam_2_l[i]
            cam_3_p = cam_3_l[i]
            cam_4_p = cam_4_l[i]
            xc_cam1 = cam_1_p.x
            yc_cam1 = cam_1_p.y
            xc_cam2 = cam_2_p.x
            yc_cam2 = cam_2_p.y
            xc_cam3 = cam_3_p.x
            yc_cam3 = cam_3_p.y
            xc_cam4 = cam_4_p.x
            yc_cam4 = cam_4_p.y

            cam_1_frame_id = obj.cam_1_frame_ids[i]
            cam_2_frame_id = obj.cam_2_frame_ids[i]
            cam_3_frame_id = obj.cam_3_frame_ids[i]
            cam_4_frame_id = obj.cam_4_frame_ids[i]

            print(cam_1_frame_id)
            print(cam_2_frame_id)
            print(cam_3_frame_id)
            print(cam_4_frame_id)
		

            f, axs = plt.subplots(1, 4, figsize=(15, 4))

            # ---------- for view 1  ----------
            ax1 = axs[0]
            image = cv2.imread(path + cam_1_frame_id)
            ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax1.scatter(xc_cam1, yc_cam1, s=8, color='blue')

            #---------- for view 2  ----------
            ax2 = axs[1]
            image = cv2.imread(path_2 + cam_2_frame_id)
            ax2.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax2.scatter(xc_cam2, yc_cam2, s=8, color='blue')

            # ---------- for view 3  ----------
            ax3 = axs[2]
            image = cv2.imread(path_3 + cam_3_frame_id)
            ax3.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax3.scatter(xc_cam3, yc_cam3, s=8, color='blue')

            #---------- for view 4  ----------
            ax4 = axs[3]
            image = cv2.imread(path_4 + cam_4_frame_id)
            ax4.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax4.scatter(xc_cam4, yc_cam4, s=8, color='blue')

            plt.savefig("/home/dissana8/LAB/"+str(obj_id)+".jpg")
            ax1.cla()
            ax2.cla()
            ax3.cla()
            ax4.cla()
        break  # we are only plotting the trajectory of the first object


if __name__ == '__main__':
    path = "~/LAB/"
    file_name = 'LAB-GROUNDTRUTH.ref'


    persons = extract_tracks_from_Annotations(path, file_name)
    print_obj_list(persons)
    #np.save("/home/dissana8/persons", persons)
 
    #===========visualising the trajectories and projections=================#
    path_1 = "/home/dissana8/LAB/Visor/cam1/"
    path_2 = "/home/dissana8/LAB/Visor/cam2/"
    path_3 = "/home/dissana8/LAB/Visor/cam3/"
    path_4 = "/home/dissana8/LAB/Visor/cam4/"
    #visualise_GT_tracks(path_1, path_2, path_3, path_4, persons)
        
    save_coords(persons, file_name[:3])
