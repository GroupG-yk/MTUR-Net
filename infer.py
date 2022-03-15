import os
import time
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms


from nets_BASE import  basic

from misc import check_mkdir
from nets_YK import  MTURNet_YKCCR

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(2019)
torch.cuda.set_device(0)

ckpt_path = './ckpt'
exp_name = 'basic'
args = {
    'snapshot': '30000',
    'depth_snapshot': ''
}

transform = transforms.Compose([
    # transforms.Resize([512,512]),
    transforms.ToTensor() ])




root = '/data1/lly/dataset/Ucolor/input_test512'

to_pil = transforms.ToPILImage()


def main():
    net = basic().cuda()

    if len(args['snapshot']) > 0:
        print('load snapshot \'%s\' for testing' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'),
                                       map_location=lambda storage, loc: storage.cuda(0)))

    net.eval()
    avg_time = []

    with torch.no_grad():

        img_list = [img_name for img_name in os.listdir(root)]

        for idx, img_name in enumerate(img_list):
            check_mkdir(
                os.path.join(ckpt_path, exp_name, '(%s) prediction_%s' % (exp_name, args['snapshot'])))
            if len(args['depth_snapshot']) > 0:
                check_mkdir(
                    os.path.join(ckpt_path, exp_name, '(%s) prediction_%s' % (exp_name, args['depth_snapshot'])))

            img = Image.open(os.path.join(root, img_name)).convert('RGB')
            w, h = img.size
            img_var = Variable(transform(img).unsqueeze(0)).cuda()

            start_time = time.time()


            res = net(img_var)

            torch.cuda.synchronize()

            # if len(args['depth_snapshot']) > 0:
            #     depth_res = depth_net(res, depth_optimize = True)

            avg_time.append(time.time() - start_time)
            ti = np.array(avg_time)
            print('predicting: %d / %d, avg_time: %.5f' % (idx + 1, len(img_list), np.mean(ti)))

            result = transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu()))


            result.save(
                os.path.join(ckpt_path, exp_name, '(%s) prediction_%s' % (
                    exp_name, args['snapshot']), img_name))



if __name__ == '__main__':
    main()
