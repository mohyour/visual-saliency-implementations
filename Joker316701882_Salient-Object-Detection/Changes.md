Training datasets needs to be downloaded as stated in the README file
Applied these changes:
- Made tensor flow 1.X.X compatible with version 2.X.X - line 19 - 24
- imread and imresize are deprecated in scipy 2.X.X. Had issues installing previous version. Used imresize from skimage.transform to replace misc.imresize(line 33, 37, 45, 49) and imread from matplotlib.pyplot to replace  misc.imread(line 29, 41)