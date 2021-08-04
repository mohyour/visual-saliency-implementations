# visual-saliency-implementations

Implementations for visual saliency

Implementation 1 - [Salient-Object-Detection](https://github.com/Joker316701882/Salient-Object-Detection) works fine

Implementation 2 - [NLDF-pytorch](https://github.com/AceCoooool/NLDF-pytorch) has error
- Running step 4 from the readme - `python demo.py --demo_img=./png/demo.jpg --trained_model=./weights/vgg16_feat.pth --cuda=True` throws error ```_pickle.UnpicklingError: A load persistent id instruction was encountered, but no persistent_load function was specified.```
Also, there seem to be some config issues between the MSRA dataset and ECSSD dataset(downloaded from link in the repo readme) in main.py from line 36. Using the ECSSD configs (line 40 - line 53) throw error and we have no access to MSRA dataset used.
