# Mask R-CNN - Camus Dataset

This notebook shows how to train, test and evaluate Mask R-CNN on [CAMUS dataset](https://www.creatis.insa-lyon.fr/Challenge/camus/).
- We note that we use a variation of this dataset where we have 4 binary masks (The forth being the union of 1 and 2 -Epicardium-) instead of one mask with the 3 classes. Check the data preprocessing notebook. We thus have the following classes corresponding to each groundtruth:  
  
    1) `Left ventricule`   
    2) `Myocardium`   
    3) `Left atrium`   
    4) `Epicardium`   

- We note that *train* section can be skipped if one wants to perform detection, evaluation... However, cells below the *detect* section are dependent of it.

Summary :
* I.   [Dataset](#dataset)
* II.  [Train](#train)
* III. [Detect](#detect)
* IV.  [Save output masks original format](#save)
* V.   [Evaluate](#evaluate)
* VI.  [Inspect detection](#inspect)
* VII. [Save segmentation's differences in **png** format](#png)  
TODO: 
- Parallel : adapt code to Pr. Bernard's code
- Evaluate on 2 classes
