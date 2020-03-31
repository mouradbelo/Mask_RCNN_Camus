"""
This scipt was only used for training

to train:
python3 samples/Miccai/Miccai.py --dataset /floyd/input/miccai_preprocess --weights custom --logs logs train >> log.txt

"""

import os
import sys
import datetime
import numpy as np
import skimage.draw
import SimpleITK as sitk
import matplotlib.pylab as plt
from imgaug import augmenters as iaa
import keras

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
from mrcnn.visualize import display_images

# Path to trained weights file
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
#if not os.path.exists(COCO_WEIGHTS_PATH):
 #   utils.download_trained_weights(COCO_WEIGHTS_PATH)
    

MICCAI16_WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs/MICCAI/mask_rcnn_miccai_0036.h5")
WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Separate mask
############################################################
from skimage import measure

def separate_mask(mask, num_pixel_treshold):
    # num_pixel_treshold = number of pixels treshold, not to take as mask for training
    all_mask = []
    # will contain all mask generated from mask
    nb_label = 0
    label_mask, nb_label = measure.label(mask, connectivity=1, return_num=True)
    #connectivity=1 for 4 neighbors, 2 for 8 neighbors
    if nb_label >= 1:
        for i in range(1,nb_label+1):
            mask_compare = np.full(np.shape(label_mask), i) 
            # check equality test and have the value i on the location of each mask
            separate_mask = np.equal(label_mask, mask_compare).astype(int) 
            # give new name to the masks
            count_non_zero = np.count_nonzero(separate_mask) 
            #print(" count non zeros : ", count_non_zero)
            if (count_non_zero > num_pixel_treshold):
                all_mask.append(separate_mask)

        if all_mask == []:
            # no mask found
            mask=np.expand_dims(mask,axis=2)
        else:
            mask = np.stack(all_mask, axis=-1)
    else:
        # no mask found
        mask=np.expand_dims(mask,axis=2)

    return mask
            
############################################################
#  Configurations
############################################################



class MiccaiConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Miccai"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 5

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + 1 Lesion

    #IMAGE_CHANNEL_COUNT = 1

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 500

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    # Image mean (RGB)
    MEAN_PIXEL = np.array([0.128, 0.128, 0.128])
    
    # BACKBONE_STRIDES for computing feature map size
    
    # Length of square anchor side in pixels
    #RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    
    IMAGE_RESIZE_MODE = "none"
    IMAGE_MIN_DIM = 336 #84
    IMAGE_MAX_DIM = 336 #84
    
    RPN_ANCHOR_RATIOS = [0.5, 1, 1.5, 2]
    
    VALIDATION_STEPS = 50
    
    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 19

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 30
    
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56) 
    
class MiccaiInferenceConfig(MiccaiConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Non-max suppression threshold to filter RPN proposals.
    RPN_NMS_THRESHOLD = 0.7


############################################################
#  Dataset
############################################################
from skimage import measure

class MiccaiDataset(utils.Dataset):
    
    def load_miccai_dataset(self, dataset_dir,subset):
        """Load a subset of the dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have 1 classe and backround is initialized per default
        self.add_class("Miccai", 1, "Lesion")
        
        assert subset in ["train", "validation", "test"]

        image_ids = []
        
        print("dataset_dir = ", dataset_dir)

        # r=root, d=directories, f = files
        for r, d, f in sorted(os.walk(dataset_dir)):#!MB to choose the same patients always for train and val, useful for transfer learning
            for file in f:    
                if '3DFLAIR' in file:
                    image_ids.append(os.path.join(r, file))

        nb_image = len(image_ids)
        
        taille_train = nb_image - 400 #int(0.6*nb_image)
        taille_val = 200 #int(0.2*nb_image)
        taille_test = 200 #nb_image - taille_train - taille_val

        if subset=="train":
            
            print("nombre image 2D : ", nb_image)
            print("taille train: ", taille_train)
            print("taille valid: ", taille_val)
            print("taille test: ", taille_test)
        
            for image_id in image_ids[:taille_train]:
                img_name = os.path.basename(image_id)
                #print(" img_name = ", img_name)
                self.add_image(
                    "Miccai",
                    image_id=img_name[len("3DFLAIR_"):len(img_name)-len(".nii.gz")],
                    #image_id[39:],
                    path=image_id)
        
        elif subset=="validation":
            for image_id in image_ids[taille_train:taille_train+taille_val]:
                img_name = os.path.basename(image_id)
                self.add_image(
                    "Miccai",
                    image_id=img_name[len("3DFLAIR_"):len(img_name)-len(".nii.gz")],
                    path=image_id)
        else:
            for image_id in image_ids[taille_train+taille_val:]:
                img_name = os.path.basename(image_id)
                self.add_image(
                    "Miccai",
                    image_id=img_name[len("3DFLAIR_"):len(img_name)-len(".nii.gz")],
                    path=image_id)
                
            
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        image_info = self.image_info[image_id]
        
        img_name = os.path.dirname(image_info['path'])
        mask_id = os.path.join(os.path.dirname(image_info['path']), "Consensus_" + 
                               str(image_info['id']) + ".nii.gz")
            
        mask=sitk.GetArrayFromImage(sitk.ReadImage(mask_id, sitk.sitkUInt8))
        
        num_pixel_treshold = 7
        mask = separate_mask(mask, num_pixel_treshold)
        
        count_non_zero = np.count_nonzero(mask) 

        if count_non_zero < num_pixel_treshold:
            # no mask
            mask = np.zeros((mask.shape[0], mask.shape[1], 1))
            class_ids = np.zeros(1, np.int32)
        
        else:
            class_ids = np.ones([mask.shape[-1]], dtype=np.int32)
        
        return mask.astype(np.bool), class_ids
        
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "Miccai":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
    
    def load_image (self, image_id):

        image=sitk.GetArrayFromImage(sitk.ReadImage(self.image_info[image_id]['path'], sitk.sitkFloat32))

        image[image < 0] = 0

        image=np.stack((image,image,image),axis=2)

        # mean pixels in all dataset :  0.13044332
        # std pixels in all dataset :  0.28245687
        
        return image


############################################################
#  Training
############################################################

def train(model, dataset_dir):
    """Train the model."""
    # Training dataset.
    dataset_train = MiccaiDataset()
    dataset_train.load_miccai_dataset(dataset_dir,"train")
    dataset_train.prepare()
    
    # Validation dataset
    dataset_val = MiccaiDataset()
    dataset_val.load_miccai_dataset(dataset_dir, "validation")
    dataset_val.prepare()
    
    # Test dataset
    #dataset_test = MiccaiDataset()
    #dataset_test.load_miccai_images(dataset_dir, "test")
    #dataset_test.prepare()

    plotter=keras.callbacks.CSVLogger(filename='training_malick.log',separator=',',append=True)
    callbacks=[plotter]
    
    augmentation = iaa.SomeOf((0, 2), [
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.OneOf([iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270)]),
                iaa.Multiply((0.8, 1.5)),
                iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])
    
    
#    print("Train conv1 layer ")
 #   model.train(dataset_train, dataset_val,
  #      learning_rate=config.LEARNING_RATE,
   #     epochs=30,
    #    custom_callbacks=callbacks,
     #   augmentation=augmentation,
      #  layers='conv1')

    print("Train heads layers")
    model.train(dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=50,
        custom_callbacks=callbacks,
        augmentation=augmentation,
        layers='heads')
    
    #print("Train 3+ layers")
    #model.train(dataset_train, dataset_val,
     #   learning_rate=config.LEARNING_RATE,
      #  epochs=45,
       # custom_callbacks=callbacks,
        #augmentation=augmentation,
        #layers='3+')

#    print("Train all layers")
 #   model.train(dataset_train, dataset_val,
  #      learning_rate=config.LEARNING_RATE/10,
   #     epochs=40,
    #    custom_callbacks=callbacks,
     #   augmentation=augmentation,
      #  layers='all')
    
    
############################################################
#  Detection
############################################################

def detect (model, image_path=None):
    
    # Run model detection 
    print("Running on {}".format(args.image))
    # Read image
    image=sitk.GetArrayFromImage(sitk.ReadImage(mask_id, sitk.sitkFloat32))
    # Detect objects
    r = model.detect([image], verbose=1)[0]
        # Save output
        #file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        #skimage.io.imsave(file_name, splash)
        
############################################################
#  Evaluation
############################################################
def evalute (model,dataset_dir):
    image_ids=dataset_val.image_ids
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, infConf,
                               image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, infConf), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
    
    print("mAP: ", np.mean(APs))
    
############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for brain s segmentation from Micccai')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' 'detect' or 'evaluate' ")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = MiccaiConfig()
    else:
        config = MiccaiInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
        flag=False
    elif args.weights.lower() == "custom":
        # Find last trained weights
        weights_path = MICCAI16_WEIGHTS_PATH
        flag=False
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights
        flag=False

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco" or flag:
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True) #, exclude=["conv1"])

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    elif args.command == "evaluate":
        evaluate(model,args.dataset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
