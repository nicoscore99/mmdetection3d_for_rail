import pickle
from os import path as osp

import mmcv
import argparse
import mmengine
import numpy as np
from mmcv.ops import roi_align
from mmdet.evaluation import bbox_overlaps
from mmengine import print_log, track_iter_progress
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

from mmdet3d.registry import DATASETS
from mmdet3d.structures.ops import box_np_ops as box_np_ops


file_name = 'gt_infos_database'

def main():
    
    print("Parsing the arguments")
    
    parser = argparse.ArgumentParser(description='Merge the gt_infos and db_infos')
    
    parser.add_argument('gt_infos', help='Path to the gt_infos')
    parser.add_argument('db_infos', help='Path to the db_infos')
    parser.add_argument('location', help='Location to save the merged data')
    
    args = parser.parse_args()
    
    # make sure we have received all the arguments
    assert args.gt_infos is not None, 'gt_infos is required'
    assert args.db_infos is not None, 'db_infos is required'
    assert args.location is not None, 'location is required'
    
    # check if the file exists
    if not osp.exists(args.gt_infos):
        raise FileNotFoundError(f'{args.gt_infos} does not exist')
    
    if not osp.exists(args.db_infos):
        raise FileNotFoundError(f'{args.db_infos} does not exist')
    
    # check if they are .pkl files
    if not args.gt_infos.endswith('.pkl'):
        raise ValueError(f'{args.gt_infos} is not a .pkl file')
    
    if not args.db_infos.endswith('.pkl'):
        raise ValueError(f'{args.db_infos} is not a .pkl file')
    
    # load the files
    with open(args.gt_infos, 'rb') as f:
        gt_infos = pickle.load(f)
        
    with open(args.db_infos, 'rb') as f:
        db_infos = pickle.load(f)
        
    # create a new dict to store the merged data
    merged_infos_train = dict()
    merged_infos_val = dict()
    
    # take 'metainfo' from db_infos
    merged_infos_train['metainfo'] = db_infos['metainfo']
    merged_infos_val['metainfo'] = db_infos['metainfo']
    
    data_list = []
    
    for key, value in gt_infos.items():
        # This iterates through the single classes
        
        ## assert that the value is of type list
        assert isinstance(value, list)
        
        data_list += value
        
    # Create a train and val split
    train_data_list = []
    val_data_list = []
    
    # Shuffle the data_list
    np.random.shuffle(data_list)
    
    # Split the data_list into train and val
    train_data_list = data_list[:int(0.8 * len(data_list))]
    val_data_list = data_list[int(0.8 * len(data_list)):]

    # print lenght information on the two datasets
    print(f'Train data list length: {len(train_data_list)}')
    print(f'Val data list length: {len(val_data_list)}')
    
    # Put into two separate lists
    merged_infos_train['data_list'] = train_data_list
    merged_infos_val['data_list'] = val_data_list
    
    # save the merged data
    with open(osp.join(args.location, f'{file_name}_train.pkl'), 'wb') as f:
        pickle.dump(merged_infos_train, f)
        
    with open(osp.join(args.location, f'{file_name}_val.pkl'), 'wb') as f:
        pickle.dump(merged_infos_val, f)
        
    print_log(f'Successfully merged the data and saved it to {file_name}_train.pkl and {file_name}_val.pkl')
    
if __name__ == '__main__':
    main()