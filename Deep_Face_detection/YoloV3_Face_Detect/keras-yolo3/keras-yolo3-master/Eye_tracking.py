import os
import argparse
from PIL import Image
import numpy as np
from yolo import YOLO, detect_video, detect_WebCam
import urllib.request as urlreq
from gaze_tracking import GazeTracking

def Eye_tracking(yolo):
    import cv2
    cam = cv2.VideoCapture(0)
    cam.set(3,1000)
    cam.set(4,600)
    gaze = GazeTracking()
    while True:
        _, frame = cam.read()
        image = Image.fromarray(frame)
        res, Faces = yolo.detect_image_with_coord(image)
        
        output = np.array(res)

        for face in Faces:
            x1, y1, x2, y2 = face

            Face_img = frame[y1:y2,x1:x2]
            gaze.refresh(Face_img)

            Ox1, Oy1, Ox2, Oy2 = gaze.annotated_frame(x1, y1)
            color = (0,0,255)
            cv2.line(output, (Ox1 - 5, Oy1), (Ox1 + 5, Oy1), color)
            cv2.line(output, (Ox1, Oy1 - 5), (Ox1, Oy1 + 5), color)
            cv2.line(output, (Ox2 - 5, Oy2), (Ox2 + 5, Oy2), color)
            cv2.line(output, (Ox2, Oy2 - 5), (Ox2, Oy2 + 5), color)

            # 60 130 165
            text = ""
            if gaze.is_blinking():
                text = "Blinking"
            elif gaze.is_right():
                text = "Looking right"
            elif gaze.is_left():
                text = "Looking left"
            elif gaze.is_center():
                text = "Looking center"
            
            left_pupil = gaze.pupil_left_relative_coords()
            right_pupil = gaze.pupil_right_relative_coords()
            cv2.putText(output, text, (x1, y1 + 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (147, 58, 31), 2)
            cv2.putText(output, "Left pupil:  " + str(left_pupil), (x1, y1 + 50), cv2.FONT_HERSHEY_DUPLEX, 0.4, (147, 58, 31), 1)
            cv2.putText(output, "Right pupil: " + str(right_pupil), (x1, y1 + 65), cv2.FONT_HERSHEY_DUPLEX, 0.4, (147, 58, 31), 1)

        cv2.imshow("Camera",output)
        if cv2.waitKey(5) == 27:
            break
    
    yolo.close_session()


if __name__ == "__main__":

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    parser.add_argument(
        "--WebCam", type=int, default=0,
        help = "Use WebCam is 1, if not 0: default is 0"
    )

    FLAGS = parser.parse_args()

    Eye_tracking(YOLO(**vars(FLAGS)))



    