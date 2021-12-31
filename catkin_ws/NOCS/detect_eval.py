'''
Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation
Detection and evaluation

Modified based on Mask R-CNN(https://github.com/matterport/Mask_RCNN)
Written by He Wang
'''

import os, sys, cv2
import time, utils
import argparse, datetime
import numpy as np
import model as Model
import _pickle as cPickle
from tqdm import tqdm
from glob import glob
from config import Config
from dataset import NOCSDataset
from train import ScenesConfig


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='detect', type=str, help='detect/eval')
parser.add_argument('--use_regression', dest='use_regression', action='store_true', default=False)
parser.add_argument('--use_delta', dest='use_delta', action='store_true', default=False)
parser.add_argument('--ckpt_path', type=str, default='logs/nocs_rcnn_res50_bin32.h5')
parser.add_argument('--data', type=str, help='val/real_test', default='real_test')
parser.add_argument('--draw', dest='draw', action='store_true', default=True,
                    help='whether draw and save detection visualization')
parser.add_argument('--num_eval', type=int, default=-1)
parser.add_argument('--gpu',  default='0', type=str)
#parser.set_defaults(draw=True)
args = parser.parse_args()

mode = args.mode
data = args.data
ckpt_path = args.ckpt_path
use_regression = args.use_regression
use_delta = args.use_delta
num_eval = args.num_eval

os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
print('Using GPU {}.'.format(args.gpu))

# Directory to save logs and trained model
MODEL_DIR = os.path.join(os.getcwd(), 'logs')


class InferenceConfig(ScenesConfig):
    '''Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    '''
    # Give the configuration a recognizable name
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    COORD_USE_REGRESSION = use_regression
    if COORD_USE_REGRESSION:
        COORD_REGRESS_LOSS = 'Soft_L1' 
    else:
        COORD_NUM_BINS = 32
    COORD_USE_DELTA = use_delta

    USE_SYMMETRY_LOSS = True
    TRAINING_AUGMENTATION = False


def img2vid(src, w=640, h=480):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid = cv2.VideoWriter(src+'.mp4', fourcc, 25, (w*2,h*2))
    for i in tqdm(sorted(glob(f'{src}/*_bbox_gt.png'))):
        bx = cv2.imread(i[:-6]+'pred.png')
        bx = np.hstack([cv2.imread(i),bx])
        i = i[:-11]+'coord_gt.png'
        cd = cv2.imread(i[:-6]+'pred.png')
        cd = np.hstack([cv2.imread(i),cd])
        im = np.vstack([bx,cd]); vid.write(im)
        cv2.imshow('nocs',im); k = cv2.waitKey(5)
        if k in (27, ord('q')): break
    vid.release(); cv2.destroyAllWindows()


if __name__ == '__main__':
    config = InferenceConfig(); config.display()
    # Training dataset
    # dataset directories
    camera_dir = os.path.join('data', 'camera')
    real_dir = os.path.join('data', 'real')
    coco_dir = os.path.join('data', 'coco')

    #  real classes
    coco_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                  'bus', 'train', 'truck', 'boat', 'traffic light',
                  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                  'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                  'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                  'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                  'kite', 'baseball bat', 'baseball glove', 'skateboard',
                  'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                  'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                  'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                  'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                  'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                  'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                  'teddy bear', 'hair drier', 'toothbrush']

    synset_names = ['BG', #0
                    'bottle', #1
                    'bowl', #2
                    'camera', #3
                    'can',  #4
                    'laptop',#5
                    'mug'#6
                    ]

    class_map = {
        'bottle': 'bottle',
        'bowl':'bowl',
        'cup':'mug',
        'laptop': 'laptop'}

    coco_cls_ids = []
    for coco_cls in class_map:
        ind = coco_names.index(coco_cls)
        coco_cls_ids.append(ind)

    assert mode in ['detect', 'eval']
    if mode == 'detect':
        # Recreate the model in inference mode
        model = Model.MaskRCNN(mode='inference', config=config, model_dir=MODEL_DIR)

        gt_dir = os.path.join('data','gts', data)
        if data == 'val':
            dataset_val = NOCSDataset(synset_names, 'val', config)
            dataset_val.load_camera_scenes(camera_dir)
            dataset_val.prepare(class_map)
            dataset = dataset_val
        elif data == 'real_test':
            dataset_real_test = NOCSDataset(synset_names, 'test', config)
            dataset_real_test.load_real_scenes(real_dir)
            dataset_real_test.prepare(class_map)
            dataset = dataset_real_test
        else:
            assert False, 'Unknown data resource.'

        # Load trained weights (fill in path to trained weights here)
        assert os.path.isfile(ckpt_path), f'There is no {ckpt_path}.'
        model.load_weights(ckpt_path, by_name=True)

        image_ids = dataset.image_ids
        save_per_images = 10
        now = datetime.datetime.now()
        save_dir = os.path.join('output', '{}_{:%Y%m%dT%H%M}'.format(data, now))
        os.makedirs(save_dir, exist_ok=True)
        
        log_file = os.path.join(save_dir, 'error_log.txt')
        f_log = open(log_file, 'w')

        if data in ['real_train', 'real_test']:
            intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
        else: ## CAMERA data
            intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]])

        elapse_times = []
        if mode != 'eval':
            for i, image_id in enumerate(image_ids):
                print('='*50)
                image_start = time.time()
                print('Image id: ', image_id)
                image_path = dataset.image_info[image_id]['path']
                print(image_path)

                # loading ground truth
                image = dataset.load_image(image_id)
                depth = dataset.load_depth(image_id)

                gt_mask, gt_coord, gt_class_ids, gt_scales, gt_domain_label = dataset.load_mask(image_id)
                gt_bbox = utils.extract_bboxes(gt_mask)
     
                result = {} # record results
                result['image_id'] = image_id
                result['image_path'] = image_path

                result['gt_class_ids'] = gt_class_ids
                result['gt_bboxes'] = gt_bbox
                result['gt_RTs'] = None
                result['gt_scales'] = gt_scales

                image_path_parsing = image_path.split('/')
                xx = f'{data}_{image_path_parsing[-2]}_{image_path_parsing[-1]}'
                gt_pkl_path = os.path.join(gt_dir, f'results_{xx}.pkl')
                print(gt_pkl_path)
                if (os.path.exists(gt_pkl_path)):
                    with open(gt_pkl_path, 'rb') as f:
                        gt = cPickle.load(f)
                    result['gt_RTs'] = gt['gt_RTs']
                    if 'handle_visibility' in gt:
                        result['gt_handle_visibility'] = gt['handle_visibility']
                        assert len(gt['handle_visibility']) == len(gt_class_ids)
                        print('got handle visibiity.')
                    else: 
                        result['gt_handle_visibility'] = np.ones_like(gt_class_ids)
                else: # align gt coord with depth to get RT
                    if not data in ['coco_val', 'coco_train']:
                        if len(gt_class_ids) == 0:
                            print('No gt instance exsits in this image.')
                        print('\nAligning ground truth...')
                        result['gt_RTs'], _, error_message, _ = utils.align(gt_class_ids, gt_mask,
                            gt_coord, depth, intrinsics, synset_names, image_path, save_dir+'/'+f'{xx}_gt_')
                        if len(error_message): f_log.write(error_message)
                    result['gt_handle_visibility'] = np.ones_like(gt_class_ids)

                ## detection
                start = time.time()
                detect_result = model.detect([image], verbose=0)
                dr = detect_result[0]
                
                print('\nDetection takes {:03f}s.'.format(time.time()-start))
                result['pred_class_ids'] = dr['class_ids']
                result['pred_bboxes'] = dr['rois']
                result['pred_RTs'] = None
                result['pred_scores'] = dr['scores']
                if len(dr['class_ids']) == 0: print('No instance is detected.')

                print('Aligning predictions...')
                result['pred_RTs'], result['pred_scales'], error_message, elapses = \
                    utils.align(dr['class_ids'], dr['masks'], dr['coords'], depth, intrinsics, 
                    synset_names, image_path) #save_dir+'/'+f'{xx}_pred_')
                elapse_times += elapses
                if len(error_message): f_log.write(error_message)

                if args.draw:
                    draw_rgb = False
                    utils.draw_detections(image, save_dir, data,
                        image_path_parsing[-2]+'_'+image_path_parsing[-1], intrinsics,
                        synset_names, draw_rgb, gt_bbox, gt_class_ids, gt_mask, gt_coord,
                        result['gt_RTs'], gt_scales, result['gt_handle_visibility'],
                        dr['rois'], dr['class_ids'], dr['masks'], dr['coords'], result['pred_RTs'],
                        dr['scores'], result['pred_scales'])

                path_parse = image_path.split('/')
                image_short_path = '_'.join(path_parse[-3:])

                save_path = os.path.join(save_dir, f'results_{image_short_path}.pkl')
                with open(save_path, 'wb') as f: cPickle.dump(result, f)
                print(f'\nResults at: {save_path}.')
                
                print('Total time: {:03f}s, Average alignment time: {:03f}s\n'.
                    format(time.time()-image_start, np.mean(elapse_times)))
            f_log.close()
    else:
        log_dir = 'output/'
        result_pkl_list = glob(os.path.join(log_dir, 'results_*.pkl'))
        result_pkl_list = sorted(result_pkl_list)[:num_eval]
        assert len(result_pkl_list)

        final_results = []
        for pkl_path in result_pkl_list:
            with open(pkl_path, 'rb') as f:
                result = cPickle.load(f)
                if not 'gt_handle_visibility' in result:
                    result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
                    print('can\'t find gt_handle_visibility in the pkl.')
                else:
                    assert len(result['gt_handle_visibility']) == len(result['gt_class_ids']), \
                        '{} {}'.format(result['gt_handle_visibility'], result['gt_class_ids'])
            if type(result) is list:
                final_results += result
            elif type(result) is dict:
                final_results.append(result)
            else:
                assert False

        aps = utils.compute_degree_cm_mAP(final_results, synset_names, log_dir,
            degree_thresholds = [5, 10, 15], #range(0, 61, 1), 
            shift_thresholds= [5, 10, 15], #np.linspace(0, 1, 31)*15, 
            iou_3d_thresholds=np.linspace(0, 1, 101),
            iou_pose_thres=0.1, use_matches_for_pose=True)

