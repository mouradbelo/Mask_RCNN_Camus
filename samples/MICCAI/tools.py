from skimage import measure
import numpy as np

def separate_mask(mask, num_pixel_treshold):
    """
        separate one mask on many instances of mask,
        remove instances with less than num_pixel_treshold pixels
    """
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

        # print(" shape of all_mask : ", np.array(all_mask).shape)
        if all_mask == []:
            # no mask found
            #print("no mask program debug !! ")
            mask=np.expand_dims(mask,axis=2)
        else:
            mask = np.stack(all_mask, axis=-1)
        # print(" shape of all_mask after stacking on last axis : ", mask.shape)
    else:
        # no mask found
        #print(" \n !! no mask !! ")
        #print(" shape of mask : ", mask.shape)
        mask=np.expand_dims(mask,axis=2)
        #print(" shape of mask after expanding dimension : ", mask.shape)
    return mask
            