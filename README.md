# KeyPointsNet
KeyPoint mini network to detect object center, inspired by center-net

input size(hw): 1024*1280；
三个文件夹中共60帧数据，以1-1,2-2文件夹中各一半数据为训练集(20张)，剩下的40张为测试集；

评估方式：
设定高度误差小于16px则检测正确，经测试，整帧精度~95%,单点精度～97%，实际误差平均8px以内；
网络运行速度：
~7ms(i7-4790 x8 RTX 2080Ti)

附件result.zip中保存测试集的检测结果和检测网络得到的HeatMap；
