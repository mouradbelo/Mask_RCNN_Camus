"""

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 camusLiteWithEva.py train --dataset=/path/to/camus/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 camus.py train --dataset=/path/to/camus/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 camus.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Run detection on an image and save result
    python3 camus.py detect --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Evaluate on validation dataset
    python3 balloon.py splash --weights=last --dataset=/path/to/camus/dataset
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
WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
VALIDATION_RATIO = 0.8

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################


class CamusConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Camus"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 6

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + 1 segment

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 600

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_MAX_INSTANCES=1
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
#     USE_MINI_MASK = True
#     MINI_MASK_SHAPE = (56, 56) 
    
class CamusInferenceConfig(CamusConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MAX_INSTANCES=1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    RPN_NMS_THRESHOLD = 0.9
    DETECTION_MIN_CONFIDENCE = 0.5


############################################################
#  Dataset
############################################################

class CamusDataset(utils.Dataset):
    
    def load_camus_images(self, dataset_dir,subset):
        """Load a subset of the dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have 1 classe and backround is initialized per default
        self.add_class("camus", 1, "left ventricule")
        
        #assert subset in ["train", "val"]
        #if subset=="train":
        #    dataset_dir = os.path.join(dataset_dir, "camus_separated")
        #else:
        #    dataset_dir = os.path.join(dataset_dir, "camus_separated")
        #Path to mhd images (not gt)
        print(dataset_dir)
        image_ids = []
        for r, d, f in os.walk(dataset_dir):
            for file in f:    
                if ('ED.mhd' in file) or ('ES.mhd' in file):
                    image_ids.append(os.path.join(r, file))
    
        numImages=len(image_ids)
        stopCount=1600#int(VALIDATION_RATIO*numImages)
        print(numImages)
        if subset=="train":
            for image_id in image_ids[:stopCount]:                
                self.add_image(
                    "camus",
                    image_id=image_id[8:27],
                    path=image_id)
        elif subset=="val":
            for image_id in image_ids[stopCount:]:                
                self.add_image(
                    "camus",
                    image_id=image_id[8:27],
                    path=image_id)
            
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask1_dir = (info['path'][:-4]+"_gt1.mhd")
        
        mask1=sitk.GetArrayFromImage(sitk.ReadImage(mask1_dir))#, sitk.sitkFloat32))
        #mask1=mask1[0,:,:]
        mask1=np.expand_dims(mask1, axis=2)
        #One mask for one class and ones array to refer to the only class_id=1
        return mask1, np.ones(mask1.shape[2], dtype=np.int32)#class_ids.astype(np.int32)
        
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "camus":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
    
    def load_image (self, image_id):
        image=sitk.GetArrayFromImage(sitk.ReadImage(self.image_info[image_id]['path']))#, sitk.sitkFloat32))
        # Empty first dimension
        image=image[0,:,:]
         # 3 channels per default, stacking for RGB
        image=np.stack((image,image,image),axis=2)
        return image
    


############################################################
#  Training
############################################################

def train(model, dataset_dir):
    """Train the model."""
    # Training dataset.
    dataset_train = CamusDataset()
    dataset_train.load_camus_images(dataset_dir,"train")
    dataset_train.prepare()
    
    # Validation dataset
    dataset_val = CamusDataset()
    dataset_val.load_camus_images(dataset_dir, "val")
    dataset_val.prepare()
    plotter=keras.callbacks.CSVLogger(filename='training.log',separator=',',append=True)
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
    
    #model.train(dataset_train, dataset_val,
     #   learning_rate=config.LEARNING_RATE,
      #  epochs=5,
       # augmentation=augmentation,
        #layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE/10,
        epochs=51,
        custom_callbacks=callbacks,
        augmentation=augmentation,
        layers='all')
    
    
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
        description='Mask R-CNN for heart segmentation')
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
        config = CamusConfig()
    else:
        config = CamusInferenceConfig()
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
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

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
