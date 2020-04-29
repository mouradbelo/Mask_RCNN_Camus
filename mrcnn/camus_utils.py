"""
Function used specifically for camus dataset
BELHAMISSI Mourad
"""

import numpy as np
from medpy import metric

# Voxel spacing in mm, here Z coordinate is not used  => pixel_spacing
voxel_spacing = [0.308,0.154]

############################################################
#   Computing evaluation metrics
############################################################


def dice (gt, segmentation, _id):
    """
    Compute the dice metric. 
    gt: Groundtruth binary masks for each class. 
    seggmentation: Dictionary result of segmentation returned by model.detect[0]. 
    returns a dictionary containing the dice value for each class
    the keys of the dictionary are the classes and the values, the dice 
    values per class.
    """
    
    # Each key of the dictionary corresponds to one of the 3 classses
    metrics = dict([
        (1,0),
        (2,0),
        (3,0),
    ])
    
    # Check if there are missing masks 
    check_missing = True
    # Make sure there are masks
    if (segmentation['masks'].ndim > 2):
        # Minimum one mask detected ?
        if (segmentation['masks'].shape[2] > 0):
            metrics[segmentation['class_ids'][0]] = metric.binary.dc(gt[:,:,segmentation['class_ids'][0]-1],                                   segmentation['masks'][:,:,0])
            check_missing = True
        # Minimum 2 masks detected ?
        if (segmentation['masks'].shape[2] > 1):
            metrics[segmentation['class_ids'][1]] = metric.binary.dc(gt[:,:,segmentation['class_ids'][1]-1],                                   segmentation['masks'][:,:,1])
            check_missing = True
        # All masks are detected ?
        if (segmentation['masks'].shape[2] > 2):
            metrics[segmentation['class_ids'][2]] = metric.binary.dc(gt[:,:,segmentation['class_ids'][2]-1],                                   segmentation['masks'][:,:,2])
            check_missing = False
        if (check_missing):
            # Show which classes were detected for the missing masks image
            print ("Only following classes detected for id ",_id,":",segmentation['class_ids'])
        return metrics
    else:
        print ("No masks detected for id: ",_id)
        pass
        
        
def assd (gt, segmentation, _id):
    """
    Compute the assymetric surface distance metric in millimeters (mm).
    gt: Groundtruth binary masks for each class.
    segmentation: Dictionary result of segmentation returned by model.detect[0]. 
    returns a dictionary containing the assd value for each class
    the keys of the dictionary are the classes and the values, the assd 
    values per class.
    """
    
    metrics = dict([
        (1,0),
        (2,0),
        (3,0),
    ])
    
    if (segmentation['masks'].ndim > 2):
        if (segmentation['masks'].shape[2] > 0):
            metrics[segmentation['class_ids'][0]] = metric.binary.assd(gt[:,:,segmentation['class_ids'][0]-1],                                 segmentation['masks'][:,:,0], voxelspacing = [voxel_spacing[1], voxel_spacing[0]])
        if (segmentation['masks'].shape[2] > 1):
            metrics[segmentation['class_ids'][1]] = metric.binary.assd(gt[:,:,segmentation['class_ids'][1]-1],                                 segmentation['masks'][:,:,1], voxelspacing = [voxel_spacing[1], voxel_spacing[0]])
        if (segmentation['masks'].shape[2] > 2):
            metrics[segmentation['class_ids'][2]] = metric.binary.assd(gt[:,:,segmentation['class_ids'][2]-1],                                 segmentation['masks'][:,:,2], voxelspacing = [voxel_spacing[1], voxel_spacing[0]])
        return metrics
    else:
        print ("No masks detected for id: ",_id)
        pass
    

def hd (gt, segmentation, _id):
    """
    Compute the hausdorff distance metric in mm.
    gt: Groundtruth binary masks for each class. 
    segmentation: Dictionary result of segmentation returned by model.detect[0]. 
    returns a dictionary containing the hd value for each class
    the keys of the dictionary are the classes and the values, the hd 
    values per class.
    """
    
    metrics = dict([
        (1,0),
        (2,0),
        (3,0),
    ])
    
    if (segmentation['masks'].ndim > 2):
        if (segmentation['masks'].shape[2] > 0):
            metrics[segmentation['class_ids'][0]] = metric.binary.hd(gt[:,:,segmentation['class_ids'][0]-1],                                   segmentation['masks'][:,:,0], voxelspacing = [voxel_spacing[1], voxel_spacing[0]])
        if (segmentation['masks'].shape[2] > 1):
            metrics[segmentation['class_ids'][1]] = metric.binary.hd(gt[:,:,segmentation['class_ids'][1]-1],                                   segmentation['masks'][:,:,1], voxelspacing = [voxel_spacing[1], voxel_spacing[0]])
        if (segmentation['masks'].shape[2] > 2):
            metrics[segmentation['class_ids'][2]] = metric.binary.hd(gt[:,:,segmentation['class_ids'][2]-1],                                   segmentation['masks'][:,:,2], voxelspacing = [voxel_spacing[1], voxel_spacing[0]])
        return metrics
    else:
        print ("No masks detected for id: ",_id)
        pass
        
############################################################
#   Normalizing bounding boxes and computing bounding boxes errors
############################################################


# We limit the detection instances to three which justifies the limit in these functions
def normalize_bbox(gt_bbox, seg):    
    """
     Get the center coordinates of gt BBox and segmentation BBox as well
     as width and height of BBox. Masks are generated in radom class order,
     for example : 2,3,1
     gt_bbox: numpy array with coordinates (y1, x1, y2, x2)
              top left corner and bottom right corner coordinates.
        
     seg: dictionary result of model.detect[0]
     returns : numpy arrays for gt and segmentation
     each array is 2 dimentional, one line per class:  (xcenter, ycenter, width, height) 
    """

    # Init with nan to skip these values when computing mean and std
    # with np.nanmean and np.nanstd
    seg_bbox_reshape = np.empty(gt_bbox.shape)
    gt_bbox_reshape = np.empty(gt_bbox.shape)
    seg_bbox_reshape[:,:] = np.nan
    gt_bbox_reshape[:,:] = np.nan
    
    # Check if there's at least one mask
    if (seg['class_ids'].ndim > 0):
        # Check if all masks are detected
        if (len(seg['class_ids']) == 3):
            # Go through the segmentation masks of each class
            j = 0
            # Go through the gt masks according to the class order of segmentation masks
            for i in seg['class_ids']:

                # (y1 + y2) / 2
                gt_bbox_reshape[j,1] = (gt_bbox [j,0] + gt_bbox [j,2]) / 2
                seg_bbox_reshape[i-1,1] = (seg['rois'][j,0] + seg['rois'][j,2]) / 2
                # (x1 + x2) / 2
                gt_bbox_reshape[j,0] = (gt_bbox [j,1] + gt_bbox [j,3]) / 2
                seg_bbox_reshape[i-1,0] = (seg['rois'][j,1] + seg['rois'][j,3]) / 2
                # (y2 - y1)
                gt_bbox_reshape[j,3] = gt_bbox [j,2] - gt_bbox [j,0]
                seg_bbox_reshape[i-1,3] = seg['rois'][j,2] - seg['rois'][j,0] 
                # (x2 - x1)
                gt_bbox_reshape[j,2] = gt_bbox [j,3] - gt_bbox [j,1]
                seg_bbox_reshape[i-1,2] = seg['rois'][j,3] - seg['rois'][j,1]
                j += 1
        # If missing masks, reshape all the gt but only the detected masks
        # nan values will remain for none detected masks
        else:
            for i in np.arange(1,4):
                gt_bbox_reshape[i-1,1] = (gt_bbox [i-1,0] + gt_bbox [i-1,2]) / 2
                gt_bbox_reshape[i-1,0] = (gt_bbox [i-1,1] + gt_bbox [i-1,3]) / 2
                gt_bbox_reshape[i-1,3] = gt_bbox [i-1,2] - gt_bbox [i-1,0]
                gt_bbox_reshape[i-1,2] = gt_bbox [i-1,3] - gt_bbox [i-1,1]
            j = 0
            for i in seg['class_ids']:
                seg_bbox_reshape[i-1,1] = (seg['rois'][j,0] + seg['rois'][j,2]) / 2
                seg_bbox_reshape[i-1,0] = (seg['rois'][j,1] + seg['rois'][j,3]) / 2
                seg_bbox_reshape[i-1,3] = seg['rois'][j,2] - seg['rois'][j,0] 
                seg_bbox_reshape[i-1,2] = seg['rois'][j,3] - seg['rois'][j,1]
                j += 1 
    else:
        print("No mask")
        pass

    return gt_bbox_reshape, seg_bbox_reshape

                                         
def compute_bbox_errors(gt_bbox, seg_bbox, class_ids, im_id):   
    """
    Compute errors between gt BBoxes and segmentation BBoxes 
    gt_bbox, seg_bbox : numpy arrays as returned by normalize_bbox
    returns a 2D numpy array containing the error for each class
    height and width erros are in millimeters
    """

    # Init with nan to skip these values when computing mean and std
    error = np.empty(gt_bbox.shape)
    error[:,:] = np.nan
    
    # Check if there is at least a mask
    if (seg_bbox.ndim > 0):
        error = abs (gt_bbox - seg_bbox)
        for i in np.arange(0,error.shape[0]):
            
            # Return coordinates in millimiters
            error[i,2] *= voxel_spacing[0]
            error[i,0] *= voxel_spacing[0]
            error[i,1] *= voxel_spacing[1]
            error[i,3] *= voxel_spacing[1]
            
            # One number precision only
            error[i,0] = "%.1f" % error[i,0]
            error[i,1] = "%.1f" % error[i,1]
            error[i,2] = "%.1f" % error[i,2]
            error[i,3] = "%.1f" % error[i,3]
            
    else: 
        pass
    return error 
