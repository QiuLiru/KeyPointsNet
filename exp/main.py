# -*-coding:utf-8-*-

from PIL import Image
from PIL import ImageChops
from PIL import ImageEnhance
import cv2
import time
import pathlib
import os
from torch import nn
import torch
import numpy as np
import inspect
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from checkpoint import save_models,restore

cfg = type('', (), {})()

cfg.use_train_for_val = False
cfg.train_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
# cfg.train_idx = range(20)
cfg.valid_idx = [idx for idx in range(60) if idx not in cfg.train_idx]
cfg.bound_limit = 4
cfg.out_range = 30
cfg.gaussian_r = 6

cfg.lr = 0.0001
cfg.weight_decay = 0
cfg.flipped = True
cfg.model_weight_path = '/home/icecola/Desktop/KeyPointsDet/model/KeyPointV3-25800.tckpt'
cfg.model_optimizer_path = '/home/icecola/Desktop/KeyPointsDet/model/adam_optimizer-25800.tckpt'


def get_pos_to_kw_map(func):
    pos_to_kw = {}
    fsig = inspect.signature(func)
    pos = 0
    for name, info in fsig.parameters.items():
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            pos_to_kw[pos] = name
        pos += 1
    return pos_to_kw


def change_default_args(**kwargs):
    def layer_wrapper(layer_class):
        class DefaultArgLayer(layer_class):
            def __init__(self, *args, **kw):
                pos_to_kw = get_pos_to_kw_map(layer_class.__init__)
                kw_to_pos = {kw: pos for pos, kw in pos_to_kw.items()}
                for key, val in kwargs.items():
                    if key not in kw and kw_to_pos[key] > len(args):
                        kw[key] = val
                super().__init__(*args, **kw)

        return DefaultArgLayer

    return layer_wrapper


class DataLoader():
    def __init__(self, config):
        self.config = config
        with open('../data/anno', 'r') as f:
            pics = [x.strip().split(" ") for x in f.readlines()]
        self.train_data = [self.load_data(pics, inx) for inx in self.config.train_idx]
        self.valid_data = [self.load_data(pics, inx) for inx in self.config.valid_idx]
        self.train_seq = range(20)

    def load_data(self, pics, inx):
        if inx < 20:
            dir = '../data/1_1'
        elif inx < 40:
            dir = '../data/2_2'
        else:
            dir = '../data/3_3'
        img = cv2.imread(os.path.join(dir, pics[inx][0] + '.bmp'), cv2.IMREAD_GRAYSCALE)
        # resize
        img_new_shape = (int(img.shape[1] / 10), int(img.shape[0] / 4))
        img_resize = cv2.resize(img, img_new_shape)
        # cv2.imshow("a",img_resize)
        # cv2.waitKey()
        w1 = int(pics[inx][1])
        h1 = int(pics[inx][2])
        w2 = int(pics[inx][3])
        h2 = int(pics[inx][4])
        label = np.array([w1, h1, w2, h2], dtype=np.float32)

        label_resize = np.array([w1 / 10, h1 / 4, w2 / 10, h2 / 4], dtype=np.float32)
        target_map = self.target_map(label_resize, img_new_shape)

        return {
            'name':pics[inx][0],
            'img': img.reshape(img.shape[0], img.shape[1], 1),
            'img_resize': img_resize.reshape(1, -1, img_resize.shape[0], img_resize.shape[1]),
            'label': label,
            'label_resize': label_resize,
            'target_map': target_map
        }

    def target_map(self, label, map_shape, radius=6):
        w1, h1, w2, h2 = label[0], label[1], label[2], label[3]
        target1 = np.array(
            [[np.exp(-((x - w1) ** 2 + (y - h1) ** 2) / (2 * radius ** 2)) for x in range(map_shape[0])] for y in
             range(map_shape[1])])
        target2 = np.array(
            [[np.exp(-((x - w2) ** 2 + (y - h2) ** 2) / (2 * radius ** 2)) for x in range(map_shape[0])] for y in
             range(map_shape[1])])
        target = target1 + target2

        # cv2.imshow("a", target)
        # cv2.waitKey()
        return target

    def getItem(self, idx, train=True):
        # normalize gray value
        if train:
            if idx == 0:
                np.random.permutation(self.train_seq)
            return self.aug_data(self.train_data[self.train_seq[idx]])
        else:
            if self.config.use_train_for_val:
                return self.train_data[idx]
            else:
                return self.valid_data[idx]

    def aug_data(self, data):
        img = data['img_resize'][0, 0, ...]
        target = data['target_map']

        # cv2.imshow("img", img)
        # cv2.waitKey()
        # clip
        if np.random.choice([True, False]):
            img_padding = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
            target_padding = cv2.copyMakeBorder(target, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
            h = np.random.randint(0, 20)
            w = np.random.randint(0, 20)
            img = img_padding[h:h + img.shape[0], w:w + img.shape[1]]
            target = target_padding[h:h + target.shape[0], w:w + target.shape[1]]

        # x-flip
        if np.random.choice([True, False]):
            img = np.fliplr(img)
            target = np.fliplr(target)

        # y-flip
        if np.random.choice([True, False]):
            img = np.fliplr(img.T).T
            target = np.fliplr(target.T).T

        # rotate
        if np.random.choice([True, False]):
            img = Image.fromarray(img, mode="L")
            target = Image.fromarray(data['target_map'])

            angle = np.random.randint(-10, 10)
            img = img.rotate(angle)
            target = target.rotate(angle)

            img = np.array(img)
            target = np.array(target)

        # cv2.imshow("img_o", img)
        # cv2.waitKey()

        return {
            'img_resize': img.reshape(1, -1, img.shape[0], img.shape[1]),
            'target_map': target,
            'label_resize': data['label_resize']
        }


class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = 'CenterNet'
        self.config = config
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.deconv1 = nn.ConvTranspose2d(64, 64, (3, 3), 2, padding=1, output_padding=1)
        self.bn_de_1 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, (3, 3), 2, padding=1, output_padding=1)
        self.bn_de_2 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 2, (1, 1))
        self.bn_5 = nn.BatchNorm2d(2)

        self.loss = nn.L1Loss()

    def forward(self, item, train=True):
        # input H W:256*128*1
        input = item[0]

        conv1 = F.relu(self.bn_1(self.conv1(input)))
        conv2 = F.relu(self.bn_2(self.conv2(conv1)))
        conv3 = F.relu(self.bn_3(self.conv3(conv2)))
        MaxP3 = self.pool3(conv3)
        conv4 = F.relu(self.bn_4(self.conv4(MaxP3)))
        MaxP4 = self.pool4(conv4)

        deconv1 = F.relu(self.bn_de_1(self.deconv1(MaxP4)))
        deconv2 = F.relu(self.bn_de_2(self.deconv2(deconv1)))

        conv5 = F.relu(self.bn_5(self.conv5(deconv2)))
        # conv5 = self.bn_5(self.conv5(deconv2))

        top_map = conv5[0, 0, ...]
        bottom_map = conv5[0, 1, ...]

        array_w = torch.as_tensor(np.array([range(top_map.shape[1])] * top_map.shape[0]), device=top_map.device,
                                  dtype=torch.float32)
        array_h = torch.as_tensor(np.array([[x] * top_map.shape[1] for x in range(top_map.shape[0])]),
                                  device=top_map.device, dtype=torch.float32)

        top_temp = top_map / torch.max(top_map)
        top_temp1 = torch.where(top_temp < 1, torch.full_like(top_temp, 0), top_temp)
        # e=torch.sum(temp)

        top_w = torch.sum(top_temp1 * array_w)  # W
        top_h = torch.sum(top_temp1 * array_h)  # H

        bottom_temp = bottom_map / torch.max(bottom_map)
        bottom_temp1 = torch.where(bottom_temp < 1, torch.full_like(bottom_temp, 0), bottom_temp)

        bottom_w = torch.sum(bottom_temp1 * array_w)  # W
        bottom_h = torch.sum(bottom_temp1 * array_h)  # H

        # pos = [top_w,top_h,bottom_w,bottom_h]
        # pos = torch.as_tensor(np.array([top_w.item(),top_h.item(),bottom_w.item(),bottom_h.item()]),device=top_map.device,dtype=torch.float32)
        pos = torch.stack((top_w, top_h, bottom_w, bottom_h), 0)

        # top_indice = torch.argmax(top_map)
        # top_pos1 = torch.as_tensor([top_indice % top_map.shape[1], top_indice / top_map.shape[1]],device=top_indice.device, dtype=torch.float32)

        '''
        min=torch.min(top_map)
        max=torch.max(top_map)
        # np.sum(np.exp(d)/np.sum(np.exp(d)) *np.array([0,1,2,3,4]))
        top_array=top_map.reshape(-1)*1
        range_ary=torch.as_tensor(torch.range(0, top_array.shape[0] - 1),device=top_array.device)
        max_position=torch.sum(torch.exp(top_array)/torch.sum(torch.exp(top_array))*range_ary)
        '''
        if train:
            label = item[1]
            '''
            b=a=np.zeros((top_map.shape[0],top_map.shape[1]))
            a[int(label[1].item())][int(label[0].item())]=1
            top_p=torch.as_tensor(a,device=top_map.device,dtype=torch.float32)
            b[int(label[3].item())][int(label[2].item())]=1
            bottom_p = torch.as_tensor(b, device=top_map.device, dtype=torch.float32)

            top_loss = torch.sum(top_map-top_p) / 100
            bottom_loss = torch.sum(bottom_map-bottom_p) / 100
            '''
            beta = 6
            h1 = int(label[1].item())
            w1 = int(label[0].item())
            target1 = torch.as_tensor(np.array(
                [[np.exp(-((x - w1) ** 2 + (y - h1) ** 2) / (2 * beta ** 2)) for x in range(top_map.shape[1])] for y in
                 range(top_map.shape[0])]), device=top_map.device, dtype=torch.float32)
            a = target1.cpu().numpy()
            cv2.imshow("a", a)
            cv2.imshow("b", np.zeros_like(a))
            cv2.waitKey()
            h2 = int(label[3].item())
            w2 = int(label[2].item())
            target2 = torch.as_tensor(np.array(
                [[np.exp(-((x - w2) ** 2 + (y - h2) ** 2) / (2 * beta ** 2)) for x in range(top_map.shape[1])] for y in
                 range(top_map.shape[0])]), device=top_map.device, dtype=torch.float32)

            # beta=self.config.beta
            # top_loss=1-torch.sum(torch.exp(top_map[(h1-beta):(h1+beta),(w1-beta):(w1+beta)]))/(torch.sum(torch.exp(top_map))+0.0000001)
            # bottom_loss=1-torch.sum(torch.exp(bottom_map[(h2-beta):(h2+beta),(w2-beta):(w2+beta)]))/(torch.sum(torch.exp(bottom_map))+0.0000001)

            # loss = (top_loss+bottom_loss)*100
            # loss = self.loss(pos[0:2], label[0:2])+self.loss(pos[2:4], label[2:4])
            loss = self.loss(top_map, target1) + self.loss(bottom_map, target2)
            return pos, loss
        else:
            return pos

    # def loss_l1(self,gt,det):

    # def optimizer(self):

    # self.optimizer=torch.optim.Adam()


class NetV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = 'KeyPoint'
        self.config = config
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.deconv1 = nn.ConvTranspose2d(64, 64, (3, 3), 2, padding=1, output_padding=1)
        self.bn_de_1 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, (3, 3), 2, padding=1, output_padding=1)
        self.bn_de_2 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 1, (1, 1))

        self.loss = nn.L1Loss()

    def forward(self, item, train=True):
        # input H W:256*128*1
        input = item[0]

        conv1 = F.relu(self.bn_1(self.conv1(input)))
        conv2 = F.relu(self.bn_2(self.conv2(conv1)))
        conv3 = F.relu(self.bn_3(self.conv3(conv2)))
        MaxP3 = self.pool3(conv3)
        conv4 = F.relu(self.bn_4(self.conv4(MaxP3)))
        MaxP4 = self.pool4(conv4)

        deconv1 = F.relu(self.bn_de_1(self.deconv1(MaxP4)))
        deconv2 = F.relu(self.bn_de_2(self.deconv2(deconv1)))

        conv5 = F.relu(self.conv5(deconv2))
        map = conv5[0, 0, ...]
        # top_indice = torch.argmax(top_map)
        # top_pos1 = torch.as_tensor([top_indice % top_map.shape[1], top_indice / top_map.shape[1]],device=top_indice.device, dtype=torch.float32)

        if train:
            loss = self.loss(map, item[1])
            return loss
        else:
            ###########
            # h1=int(item[1][1].item())
            # w1=int(item[1][0].item())
            # ctr1=(w1,h1)
            #
            # h2=int(item[1][3].item())
            # w2=int(item[1][2].item())
            # ctr2=(w2,h2)
            #
            # fake_pred=get_fake_pred(ctr1,ctr2,map.shape,radius=self.config.gaussian_r)
            # map=torch.as_tensor(fake_pred,dtype=torch.float,device=map.device)
            ###########
            map_array = map.reshape(-1)
            map_ar = map_array.cpu().detach().numpy()
            map_idx = np.argsort(map_ar)[::-1]
            pos1 = [map_idx[0] % map.shape[1], map_idx[0] // map.shape[1]]
            for x in map_idx[1:]:
                pos2 = [x % map.shape[1], x // map.shape[1]]
                if np.abs(pos2[1] - pos1[1]) > self.config.out_range:
                    break

            pos = torch.as_tensor(np.hstack((pos1, pos2)), device=map.device, dtype=torch.float32)

            return pos, map.cpu().detach().numpy()

    def predict(self, map):
        pass

class NetV3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.name = 'KeyPointV3'
        self.config = config
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.deconv1 = nn.ConvTranspose2d(128, 128, (3, 3), 2, padding=1, output_padding=1)
        self.bn_de_1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(256, 64, (3, 3), 2, padding=1, output_padding=1)
        self.bn_de_2 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 1, (1, 1))

        self.loss = nn.L1Loss()

    def forward(self, item, train=True):
        # input H W:256*128*1
        input = item[0]

        conv1 = F.relu(self.bn_1(self.conv1(input)))
        conv2 = F.relu(self.bn_2(self.conv2(conv1)))
        conv3 = F.relu(self.bn_3(self.conv3(conv2)))
        MaxP3 = self.pool3(conv3)
        conv4 = F.relu(self.bn_4(self.conv4(MaxP3)))
        MaxP4 = self.pool4(conv4)

        deconv1 = F.relu(self.bn_de_1(self.deconv1(MaxP4)))
        cat1=torch.cat((deconv1,conv4),1)

        deconv2 = F.relu(self.bn_de_2(self.deconv2(cat1)))
        #cat2 = torch.cat((deconv2, conv3), 1)

        conv5 = F.relu(self.conv5(deconv2))
        map = conv5[0, 0, ...]
        # top_indice = torch.argmax(top_map)
        # top_pos1 = torch.as_tensor([top_indice % top_map.shape[1], top_indice / top_map.shape[1]],device=top_indice.device, dtype=torch.float32)

        if train:
            loss = self.loss(map, item[1])
            return loss
        else:
            ###########
            # h1=int(item[1][1].item())
            # w1=int(item[1][0].item())
            # ctr1=(w1,h1)
            #
            # h2=int(item[1][3].item())
            # w2=int(item[1][2].item())
            # ctr2=(w2,h2)
            #
            # fake_pred=get_fake_pred(ctr1,ctr2,map.shape,radius=self.config.gaussian_r)
            # map=torch.as_tensor(fake_pred,dtype=torch.float,device=map.device)
            ###########
            map_array = map.reshape(-1)
            map_ar = map_array.cpu().detach().numpy()
            map_idx = np.argsort(map_ar)[::-1]
            pos1 = [map_idx[0] % map.shape[1], map_idx[0] // map.shape[1]]
            for x in map_idx[1:]:
                pos2 = [x % map.shape[1], x // map.shape[1]]
                if np.abs(pos2[1] - pos1[1]) > self.config.out_range:
                    break

            pos = torch.as_tensor(np.hstack((pos1, pos2)), device=map.device, dtype=torch.float32)

            return pos, map.cpu().detach().numpy()

    def predict(self, map):
        pass


class TrainProcessor():
    def __init__(self, config):
        self.config = config
        self.data = DataLoader(config)
        self.net = NetV3(config)
        root_dir = pathlib.Path('./../')
        log_dir_name = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_dir = root_dir / 'summary' / log_dir_name
        self.eval_checkpoint_dir = log_dir / 'eval_checkpoint'
        self.img_box_dir = log_dir / 'img_box'
        log_dir.mkdir(parents=True, exist_ok=True)
        self.eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(log_dir))
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.optimizer.name = 'adam_optimizer'
        try:
            restore(config.model_weight_path,self.net)
            restore(config.model_optimizer_path,self.optimizer)
        except :
            print('Training the model from origin')

    def convert_tensor(self, tensor):
        device = torch.device("cuda:0")
        img_tsr = torch.as_tensor(tensor['img_resize'] / 255, dtype=torch.float32, device=device)
        target_map = torch.as_tensor(np.ascontiguousarray(tensor['target_map'], dtype=np.float32), dtype=torch.float32,
                                     device=device)
        label_tsr = torch.as_tensor(tensor['label_resize'], dtype=torch.float32, device=device)
        return (img_tsr, target_map, label_tsr)

    def junge(self, pos, label):
        bound = self.config.bound_limit
        single_acc = 0
        a = torch.abs(pos[0] - label[0]) < bound and torch.abs(pos[1] - label[1]) < bound
        b = torch.abs(pos[0] - label[2]) < bound and torch.abs(pos[1] - label[3]) < bound
        c = torch.abs(pos[2] - label[0]) < bound and torch.abs(pos[3] - label[1]) < bound
        d = torch.abs(pos[2] - label[2]) < bound and torch.abs(pos[3] - label[3]) < bound

        error = 0.0
        error1 = error2 = 0.0
        if a:
            error1 = torch.sqrt((pos[0] - label[0]) ** 2 + (pos[1] - label[1]) ** 2)
        elif b:
            error1 = torch.sqrt((pos[0] - label[2]) ** 2 + (pos[1] - label[3]) ** 2)
        if c:
            error2 = torch.sqrt((pos[2] - label[0]) ** 2 + (pos[3] - label[1]) ** 2)
        elif d:
            error2 = torch.sqrt((pos[2] - label[2]) ** 2 + (pos[3] - label[3]) ** 2)

        if (a and d) or (b and c):
            single_acc = 2
            error = error1 + error2
        else:
            if a or b:
                single_acc = 1
                error = error1
            elif c or d:
                single_acc = 1
                error = error2

        # if (a and d) or (b and c):
        #     single_acc = 2
        # else:
        #     if a or b or c or d:
        #         single_acc = 1

        return single_acc, error

    def draw_result(self, pos, item, dir):
        w1=pos[0]*10
        h1=pos[1]*4
        w2=pos[2]*10
        h2=pos[3]*4

        img=np.concatenate((item['img'],item['img'],item['img']),2)
        name=item['name']

        img_box=cv2.circle(img,(w1,h1), 5, (0,0,255), -1)
        img_box = cv2.circle(img_box, (w2, h2), 5, (0, 0, 255), -1)

        pic_name = os.path.join(dir, name +'_result.bmp')
        cv2.imwrite(pic_name, img_box)

    def draw_feature_map(self,feature,input,save_dir):
        feature=feature.squeeze()
        name=input['name']
        origin_img=input['img_resize'].squeeze()[..., np.newaxis]
        origin_img_3C=np.concatenate((origin_img,origin_img,origin_img),axis=2)
        combination=feature*0.5+origin_img_3C*0.5
        pic_name = os.path.join(save_dir, name +'_featureMap.bmp')
        cv2.imwrite(pic_name, combination)
        # cv2.imshow('a+b',combination)
        # cv2.waitKey()

    def training(self):
        self.net.cuda()
        global_step = 0
        for epoch in range(2000):
            self.net.train()
            self.optimizer.zero_grad()
            if epoch==0:
                train_length=0
            else:
                train_length=20
            for idx in range(train_length):  # 20
                item = self.data.getItem(idx, train=True)
                item_cuda = self.convert_tensor(item)

                loss = self.net(item_cuda)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if idx % 10 == 0:
                    self.writer.add_scalar('loss', loss, global_step)
                    print('global_step: %5d ,  loss: %.6f' % (global_step, loss.item()))
                global_step += 1

            if (epoch) % 30 == 0:
                pass
                save_models(self.eval_checkpoint_dir, [self.net, self.optimizer], global_step, max_to_keep=1000)

            if (epoch) % 30 == 0:
                self.net.eval()
                #### eval #####
                correct = 0.00
                single_correct = 0.0
                total_error=0.0

                number = 20 if self.config.use_train_for_val else 40
                for idx in range(number):  #
                    val_item = self.data.getItem(idx, train=False)
                    val_item_cuda = self.convert_tensor(val_item)
                    t=time.time()
                    val_pos, map_np = self.net(val_item_cuda, train=False)
                    cost=time.time()-t
                    # ---------draw image in tensor-board and save ------------
                    map_norm = np.zeros_like(map_np[..., np.newaxis])
                    cv2.normalize(map_np[..., np.newaxis], map_norm, alpha=1, beta=0, norm_type=cv2.NORM_MINMAX)
                    map_norm_255=(map_norm* 255).astype(np.uint8)
                    map_norm_255_jet = cv2.applyColorMap(map_norm_255, colormap=cv2.COLORMAP_JET)[np.newaxis, ...]
                    if True:
                        image_out_dir = pathlib.Path(os.path.join(self.img_box_dir, str(epoch)))
                        image_out_dir.mkdir(parents=True, exist_ok=True)
                        self.draw_result(val_pos, val_item, str(image_out_dir))
                        self.draw_feature_map(map_norm_255_jet,val_item,str(image_out_dir))
                    self.writer.add_images("EvalMapIndex_" + str(idx), map_norm_255_jet, global_step=global_step, dataformats='NHWC')
                    # ----------------------------------------------
                    val_label = val_item_cuda[2]
                    single_acc, error = self.junge(val_pos, val_label)
                    if single_acc > 0:
                        single_correct = single_correct + single_acc
                        total_error+=error
                        if single_acc == 2:
                            correct += 1

                single_accuracy = single_correct / (number * 2)
                accuracy = correct / number
                total_error=total_error/(single_correct+1e-5)

                self.writer.add_scalar('accuracy', accuracy, global_step)
                self.writer.add_scalar('single_accuracy', single_accuracy, global_step)
                self.writer.add_scalar('total_error', total_error, global_step)
                print('Eval epoch %3d,  accuracy: %.6f,  single_accuracy: %.6f   total_error:  %.6f\n   ' % (
                    (epoch + 1), accuracy, single_accuracy, total_error))


if __name__ == "__main__":
    train = TrainProcessor(cfg)
    train.training()
    '''
    cv2.namedWindow("ResizeWindows", cv2.WINDOW_AUTOSIZE)
    dir='/home/icecola/Desktop/KeyPointsDet/data/1_1'
    pics=sorted(os.listdir(dir))
    for pic in pics:
        pic_name=os.path.join(dir,pic)
        img=cv2.imread(pic_name,cv2.IMREAD_GRAYSCALE)
        img_new_shape=(int(img.shape[0]/10),int(img.shape[1]/4))
        new_img=cv2.resize(img,img_new_shape)
        cv2.imshow("ResizeWindows",new_img)
        cv2.waitKey()
    '''
