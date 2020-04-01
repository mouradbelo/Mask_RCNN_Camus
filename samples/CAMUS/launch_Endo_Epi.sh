floyd init ge-insa-lyon/projects/project-5-unet-vs-mask-rcnn
floyd run --gpu --env tensorflow-1.14 --data ge-insa-lyon/datasets/camus_separated/1:input --data ge-insa-lyon/projects/project-5-unet-vs-mask-rcnn/159:/model 'python3 setup.py install && python3 samples/camus/camusNoLA.py --dataset /input --weights /model/logs/camus20200212T1544/mask_rcnn_camus_0029.h5 --logs logs train'

