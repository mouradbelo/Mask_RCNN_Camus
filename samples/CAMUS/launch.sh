floyd init ge-insa-lyon/projects/project-5-unet-vs-mask-rcnn
floyd run --gpu --env tensorflow-1.14 --data ge-insa-lyon/datasets/camus-separated/1:input --data ge-insa-lyon/projects/project-5-unet-vs-mask-rcnn/76:/model 'python3 setup.py install && python3 samples/camus/camusWithEva.py --dataset /input --weights /model/logs/camus20191201T1936/mask_rcnn_camus_0055.h5 --logs logs train'
