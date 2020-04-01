# Mask R-CNN - Camus Dataset

This repo uses Mask R-CNN on [CAMUS dataset](https://www.creatis.insa-lyon.fr/Challenge/camus/).
- We note that we use a variation of this dataset where we have 4 binary masks (The forth being the union of 1 and 2 -Epicardium-) instead of one mask with the 3 classes. Check the data preprocessing notebook. We thus have the following classes corresponding to each groundtruth:  
  
    1) `Left ventricule`   
    2) `Myocardium`   
    3) `Left atrium`   
    4) `Epicardium`   
    
Notebooks and train/test/evaluate python files are in `samples/CAMUS`.  
Scripts used for CAMUS are in `mrcnn/camus_scripts.py`

Summary :
* I.   [Dataset](#dataset)
* II.  [Train](#train)
* III. [Detect](#detect)
* IV.  [Save output masks original format](#save)
* V.   [Evaluate](#evaluate)
* VI.  [Inspect detection](#inspect)
* VII. [Save segmentation's differences in **png** format](#png)  

## Installation
1. Clone this repository
2. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
3. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ``` 
4. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases).  CAMUS weights will be uploaded soon to be downloaded.

