# KeyPointsNet

 KeyPoint mini network to detect object center, inspired by center-net
 
### DataSet
There are 3 folders each containing 20 frames of data;Training dataset is consisted with half of ` 1-1` and `2-2` , totally 20 frames. Validation dataset is consisted with the rest, totally 40 frames. 
### Running Speed
7ms (On RTX 2080Ti & i7 4790 x8)
 
### Eval metric
Set the error bound within 16px of height(y) and 40px of width(x) based on the input picture size, 1024*1280.

### Eval Result
Single Point Accuracy:97%
Frame Accuracy:95%
 
### demo
[](FinalResult/Result-qlr-20190607/result_pic/20181125113025019_flipped_featureMap.bmp)
