#!/usr/bin/python3
# -*- coding:utf-8 -*-
import sys

sys.path.append('/workspace/hugo_py')

import cv2
import numpy as np
import time
from ctypes import *
from darknet import  detect_image
from darknet import  load_net_custom
from darknet import  load_meta
from darknet import  IMAGE
from darknet import network_width, network_height
from camera import JetCamera
import traceback 

cap_w = 640
cap_h = 360
cap_fps = 10


class DarkNetWrap(object):
    def __init__(self, config_path='', weight_path='', meta_path=''):
        self.net = load_net_custom(config_path.encode('utf-8'), weight_path.encode('utf-8'), 0, 1)
        self.meta = load_meta(meta_path.encode('utf-8'))
        self.thresh = .5
        self.hier_thresh = .5
        self.nms = .45

    def detect(self, img):
        image_list = [img]

        pred_height, pred_width, c = image_list[0].shape
        net_width, net_height = (network_width(self.net), network_height(self.net))
        img_list = []

        for custom_image_bgr in image_list:
            custom_image = cv2.cvtColor(custom_image_bgr, cv2.COLOR_BGR2RGB)
            custom_image = cv2.resize(
            custom_image, (net_width, net_height), interpolation=cv2.INTER_NEAREST)
            custom_image = custom_image.transpose(2, 0, 1)
            img_list.append(custom_image)

        arr = np.concatenate(img_list, axis=0)
        arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
        data = arr.ctypes.data_as(POINTER(c_float))
        im = IMAGE(net_width, net_height, c, data)


        ret_lst = detect_image(self.net, self.meta, im, self.thresh, self.hier_thresh, self.nms)
        ret_out_lst = []

        if ret_lst:
            for ret in ret_lst:
                x, y, w, h =  ret[2]
                x = x * pred_width / net_width
                y = y * pred_height / net_height 
                w = w * pred_width / net_width 
                h = h * pred_height / net_height 

                ret = (ret[0], ret[1], 
                    (int(x - w / 2) , int(y - h / 2),
                     int(x + w / 2) , int(y + h / 2)))

                ret_out_lst.append(ret)
                 
        return ret_out_lst


def main():
    cam = JetCamera(cap_w, cap_h, cap_fps)
    fd = DarkNetWrap(config_path='yoloface-500k-v2.cfg',
                      weight_path='yoloface-500k-v2.weights',
                      meta_path='face.data'
                                          )
    if not cam:
        return
    print(cam.cap_str)
    cam.open()

    cnt = 0
    while True:
        try:
            ret, frame = cam.read()
            #print("camera read one frame ")
            if not ret:
                break

            t0 = time.time()
            res = fd.detect(frame)
            t1 = time.time()

            cnt += 1

            if cnt % 100 == 0:
                print("frame cnt [%d] yoloface detect delay = %.1fms" % (cnt, (t1 - t0) * 1000))

            for ret in res:
                r = ret[2]
                #print("ret = %s, %s" % (ret, r))
                cv2.rectangle(frame, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (255, 255, 0))

            cv2.imshow('haha', frame)
            cv2.waitKey(1)
        except:
            traceback.print_exc()
            break 

    cam.close()


if __name__ == '__main__':
    main()
