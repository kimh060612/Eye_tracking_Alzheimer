import os
import errno
import math
from PIL import Image

topdir = "/home/michael/FDDB"

# in
fddb_annotations = topdir +'/FDDB-annotations/Face-annotation.txt'
fddb_paths = topdir +'/FDDB-folds/FDDB-paths.txt'

# out
fddb_absolute_paths = topdir+'/fddb.paths'
fddb_classes_file = topdir+'/fddb.names'
fddb_config_file = topdir+'/fddb.data'

###############################
# individual annotation files #
###############################

total = []
total_path = "/home/michael/Desktop/Eye_tracking/Deep_Face_detection/YoloV3_Face_Detect/keras-yolo3/keras-yolo3-master/data_access/Face_annotation.txt"

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    finally:
        return path
        
with open(fddb_annotations, 'r') as annotations:
    for filepath in annotations:
        if filepath == "\n" :
            pass
        else :
            # make labels/<year>/<month>/<day>/big directory tree
            filepath_split = filepath.rsplit('/')
            filepath_dir = make_sure_path_exists(topdir + '/labels/' + '/'.join(filepath_split[0:len(filepath_split)-1]))
            # image annotation filepath
            filepath_clean = topdir+'/labels/'+filepath.rstrip('\n')
            filePath_save = topdir+'/originalPics/'+filepath.rstrip('\n')
            # make annotation
            tmp = ""
            for bbox in range(int(next(annotations))):
                # supplied values
                current_line = next(annotations)
                current_line_split = current_line.split()
                major_axis_radius = float(current_line_split[0])
                minor_axis_radius = float(current_line_split[1])
                angle = float(current_line_split[2])
                center_x = float(current_line_split[3])
                center_y = float(current_line_split[4])
                # find image dimensions
                img_width = 0
                img_height = 0
                with Image.open(filepath_clean.replace('labels','originalPics') + '.jpg') as img:
                    img_width, img_height = img.size
                # calculate bounding box of rotated ellipse
                calc_x = math.sqrt(major_axis_radius**2 * math.cos(angle)**2 + minor_axis_radius**2 * math.sin(angle)**2)
                calc_y = math.sqrt(major_axis_radius**2 * math.sin(angle)**2 + minor_axis_radius**2 * math.cos(angle)**2)
                # 1 class
                label = 0
                # bounding box
                bbox_x = center_x - calc_x 
                bbox_y = center_y - calc_y 
                bbox_w = (2 * calc_x)
                bbox_h = (2 * calc_y)
                if bbox_x > 0 and bbox_y > 0:
                    tmp += ' {},{},{},{},{}'.format(int(bbox_x), int(bbox_y), int(bbox_w + bbox_x), int(bbox_h + bbox_y), label)
                else :
                    pass
            tmp = filePath_save+".jpg" + tmp + "\n"
            total.append(tmp)

with open(total_path, "w") as File:
    File.writelines(total)

with open(fddb_classes_file, 'w') as classes:
    classes.write('0\n1')
