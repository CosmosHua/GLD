#!/usr/bin/python3
# coding: utf-8

import sys, cv2
import numpy as np
import pyzed.sl as sl
from tqdm import trange
from pathlib import Path
from glob import glob


# sl.VIEW.SIDE_BY_SIDE = (sl.VIEW.LEFT, sl.VIEW.RIGHT)
##########################################################################################
def svo2img(svo, dst=None, gap=1, mod=3):
    assert mod in (0,1,2,3); print("Process %s:"%svo)
    if mod==0: k,view = (1,),  (sl.VIEW.LEFT,)
    if mod==1: k,view = (2,),  (sl.VIEW.SIDE_BY_SIDE,)
    if mod==2: k,view = (1,1), (sl.VIEW.DEPTH, sl.VIEW.LEFT)
    if mod==3: k,view = (1,2), (sl.VIEW.DEPTH, sl.VIEW.SIDE_BY_SIDE)
    dst = Path(dst if dst else svo[:-4]); dst.mkdir(exist_ok=True)

    # Specify SVO path parameter
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo)
    init_params.svo_real_time_mode = False  # Don't convert in realtime
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # millimeters for depth

    # Open the SVO file specified as a parameter
    zed = sl.Camera() # Create ZED objects
    assert zed.open(init_params) == sl.ERROR_CODE.SUCCESS
    
    # Get image size, for depth_left_right side-by-side image
    image_size = zed.get_camera_information().camera_resolution
    height, width = image_size.height, image_size.width
    
    # Prepare image container, equivalent to CV_8UC4
    RGBA = np.zeros((height, width*sum(k), 4), dtype=np.uint8)
    img_mat = sl.Mat() # Prepare single image container
    
    rt_param = sl.RuntimeParameters()
    #rt_param.sensing_mode = sl.SENSING_MODE.FILL
    NF = zed.get_svo_number_of_frames()
    """while zed.grab(rt_param)==sl.ERROR_CODE.SUCCESS:
        i = zed.get_svo_position()
        if i%gap != 0: continue
        progress_bar((i+1) / NF) # Display progress
        rgbd = str(dst/("%s_DLR%06d.jpg"%(str(dst),i)) )
        dp16 = str(dst/("%s_Depth%06d.png"%(str(dst),i)) )
        
        # Save Depth_Left_Right images
        for j,v in enumerate(view): # Retrieve SVO images
            zed.retrieve_image(img_mat, v)
            a = sum(k[:j]); b = a+k[j]
            RGBA[:, a*width:b*width, :] = img_mat.get_data()
        cv2.imwrite(rgbd, cv2.cvtColor(RGBA,cv2.COLOR_RGBA2RGB))

        # Save depth images (convert to uint16)
        if sl.VIEW.DEPTH not in view: continue
        zed.retrieve_measure(img_mat, sl.MEASURE.DEPTH)
        cv2.imwrite(str(dp16), img_mat.get_data().astype(np.uint16))
    sys.stdout.write("\nSVO end has been reached.\n")"""
    for i in trange(NF, ascii=True):
        assert zed.grab(rt_param)==sl.ERROR_CODE.SUCCESS
        if i%gap != 0: continue
        rgbd = str(dst/("%s_DLR%06d.jpg"%(str(dst),i)) )
        dp16 = str(dst/("%s_Depth%06d.png"%(str(dst),i)) )
        
        # Save Depth_Left_Right images
        for j,v in enumerate(view): # Retrieve SVO images
            zed.retrieve_image(img_mat, v)
            a = sum(k[:j]); b = a+k[j]
            RGBA[:, a*width:b*width, :] = img_mat.get_data()
        cv2.imwrite(rgbd, cv2.cvtColor(RGBA,cv2.COLOR_RGBA2RGB))

        if sl.VIEW.DEPTH not in view: continue
        # Save depth images (convert to uint16)
        zed.retrieve_measure(img_mat, sl.MEASURE.DEPTH)
        cv2.imwrite(dp16, img_mat.get_data().astype(np.uint16))
    zed.close()


def progress_bar(percent, bar_length=50):
    done = int(bar_length * percent)
    done = '=' * done + '-' * (bar_length - done)
    sys.stdout.write('[%s] %.2f%s\r' % (done, percent*100, '%'))
    sys.stdout.flush()


##########################################################################################
if __name__ == "__main__":
    dst = sys.argv[2] if len(sys.argv)>2 else None
    src = Path(sys.argv[1] if len(sys.argv)>1 else ".")
    src = [str(src)] if src.is_file() else glob(str(src/"*.svo"))
    for i in src: svo2img(i, dst, gap=15, mod=3)
