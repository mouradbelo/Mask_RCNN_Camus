DEBUG=False
class CamusDataset(utils.Dataset):    
     def load_camus_images(self, dataset_dir,subset,test=0):
        """Load a subset of the dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        
        ratio_VALIDATION=0.8
        # Add classes. We have 1 classe and backround is initialized per default
        self.add_class("camus", 1, "left ventricule")
        if(test):
            dataset_dir = os.path.join(dataset_dir, "camus_test")
        else:
            assert subset in ["train", "validation"]
            if subset=="train":
                dataset_dir = os.path.join(dataset_dir, "camus_separated")
            else:
                dataset_dir = os.path.join(dataset_dir, "camus_separated")
        #Path to mhd images (not gt)
        print(dataset_dir)
        image_ids = []
        # r=root, d=directories, f = files
       
        
        for r, d, f in os.walk(dataset_dir):
            for file in f:    
                if ('ED.mhd' in file) or ('ES.mhd' in file):
                    image_ids.append(os.path.join(r, file))
        numImages=len(image_ids)
        
        #melanger les donnes
        #np.random.shuffle(image_ids)
        #
        stopCount=int(ratio_VALIDATION*numImages)
        print(numImages)
        if subset=="train":
            for image_id in image_ids[:stopCount]:                
                self.add_image(
                "camus",
                image_id=image_id[29:48],
                path=image_id)
        elif subset=="validation":
            for image_id in image_ids[stopCount:]:                
                self.add_image(
                "camus",
                image_id=image_id[29:48],
                path=image_id)
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        #mask1_dir = os.path.join(os.path.dirname(info['path']),info['id']+"_gt.mhd")
        mask1_dir = (info['path'][:-4]+"_gt1.mhd")
        
        mask1=sitk.GetArrayFromImage(sitk.ReadImage(mask1_dir, sitk.sitkFloat32))
        #mask1=mask1[0,:,:]
        if DEBUG:
            print(dataset_dir)
            print("mask direction"+str(mask1_dir))
            print("path"+str(info['path']))
            print("info"+str(info['id']))
            print("shape"+str(mask1.shape))
        mask1=np.expand_dims(mask1, axis=2)
        return mask1, np.arange(1,2, dtype=np.int32)#class_ids.astype(np.int32)  
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "camus":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
    
    def load_image (self, image_id):
        image=sitk.GetArrayFromImage(sitk.ReadImage(self.image_info[image_id]['path'], sitk.sitkFloat32))
        image=image[0,:,:]
        image=np.stack((image,image,image),axis=2)
        #print("shape",image.shape)
        return image
    
