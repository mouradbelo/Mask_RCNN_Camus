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

