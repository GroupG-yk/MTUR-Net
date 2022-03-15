clear all;
clc;

% ground truth
GTfolder = 'G:\program\paper\Ucolor_final_model_corrected\gt_test512\';
%GTfolder = '/home/xwhu/dataset/Rain100H/test/gt/';


RESfolder = 'G:\program\paper\Ucolor_final_model_corrected\DGNLNet_ableation_add_prediction_30000\';
% DGNLNet_ablation_with_Sigmoid_prediction_30000 PSNR=22.3629, ssim=0.895912.
% DGNLNet_ableation_add_prediction_30000 PSNR=22.1346, ssim=0.895855.
% DGNLNet_YKCCR_prediction_30000    PSNR=22.6039, ssim=0.900806
% DGNLNet_YKCCR_prediction_40000    PSNR=21.9718, ssim=0.890398
% DGNLNet_UW_BASE_prediction_30000  PSNR=22.1707, ssim=0.889254
test_input = textread('G:\program\paper\Ucolor_final_model_corrected\list\outputHU.txt', '%s');
test_gt = textread('G:\program\paper\Ucolor_final_model_corrected\list\test_gt.txt', '%s');

%test_input = textread('/home/xwhu/dataset/Rain100H/test.txt', '%s');
%test_gt = test_input;

% % raincity
% GTfolder = 'E:\dataset\cityscapes\leftImg8bit_trainval_rain\image\val/';
% RESfolder = 'E:\dataset\cityscapes\leftImg8bit_trainval_rain\DGNLNet_test_prediction_40000/';
% 
% test_input = textread('E:\Program\DAF-Net-master\data\RainCityscapes/test_input.txt', '%s');
% test_gt = textread('E:\Program\DAF-Net-master\data\RainCityscapes/test_gt.txt', '%s');

k=0;
for i = 1:length(test_gt)
    
    strGT = [GTfolder, test_gt{i}];
    %strRES = [RESfolder, test_input{i}];
    
    strRES = [RESfolder, test_input{i}(1:end-4) '.png'];
    
    
%     GT = im2double(imread(strGT));
%     RES = im2double(imread(strRES));

    if exist(strRES)
        k = k+1;
    else
        continue;
    end
    
    image_gt = imread(strGT);
    image_res = imread(strRES);
    
    if sum(size(image_res)==[512, 512, 3])~=3
       image_res = imresize(image_res,[512, 512]);
    end
    
    if sum(size(image_gt)==[512, 512, 3])~=3
       image_gt = imresize(image_gt,[512, 512]);
    end
    
    GT = im2double(image_gt);
    RES = im2double(image_res);
    
    value_psnr_s6(k) = psnr(RES, GT);
    value_mssim_s6(k)  = ssim(RES*255, GT*255);
    fprintf('idx=%u, PSNR=%g, ssim=%g.\n', k, value_psnr_s6(k), value_mssim_s6(k));
    
end% 
% 
fprintf('Mean: PSNR=%g, ssim=%g.\n', mean(value_psnr_s6(:)), mean(value_mssim_s6(:)));

%Mean: PSNR=27.986, ssim=0.898293.



