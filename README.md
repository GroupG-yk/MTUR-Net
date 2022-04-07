# MTUR-Net
## Medium Transmission Map Matters for Learning to Restore Real-world Underwater Images
by Kai Yan, Lanyue Liang, Ziqiang zhen, GUoqing Wang and Yang Yang  
Here is the open source code of the MTUR-Net.If you use our code, please consider citing our paper. Thanks.  

### Dataset
https://li-chongyi.github.io/proj_benchmark.html

1.For the MT dataset and data lists:

Baidu Cloud:     https://pan.baidu.com/s/1TYkjOSZlRWNaEwckis3rtQ    password: mtur 


### Test


1.Test the MTUR-Net:
If you want to use our pretrained model, please load 30000.pth in infer.py.
```
python infer.py
```
### Train

1.Test the MTUR-Net:

```
python train.py
```
### For evaluation
PSNR, SSIM : evaluate.m
UIQM, UCIQE :
https://github.com/xahidbuffon/FUnIE-GAN/tree/master/Evaluation
