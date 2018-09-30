from flask import Flask
from flask import request
from flask import jsonify

application = Flask(__name__)

import cv2
import numpy as np
import lanms
import json


def merge_boxes(boxes, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param boxes: boxes
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    boxes = np.asarray(boxes, dtype=np.float32)
    boxes = np.reshape(boxes, (-1, 9))

    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)

    if boxes.shape[0] == 0:
        return None

    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


@application.route("/detect/", methods=['GET', 'POST'])
def detect():
    boxes = request.form['boxes']
    boxes = boxes.split(',')
    boxes.pop()
    print(len(boxes))
    w = float(request.form['w'])
    h = float(request.form['h'])
    boxes = list([float(a) for a in boxes])
    ratio_w = float(request.form['ratio_w'])
    ratio_h = float(request.form['ratio_h'])

    boxes = merge_boxes(boxes)
    if boxes is not None:
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h

    ans = []
    if boxes is not None:
        for box in boxes:
            # to avoid submitting errors
            box = sort_poly(box.astype(np.int32))
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                continue

            box[0, 0] -= 0.05 * w * 2
            box[0, 1] -= 0.05 * h * 2
            box[1, 0] += 0.05 * w * 2
            box[1, 1] -= 0.05 * h * 2
            box[2, 0] += 0.05 * w * 2
            box[2, 1] += 0.05 * h * 2
            box[3, 0] -= 0.05 * w * 2
            box[3, 1] += 0.05 * h * 2

            coordinates = (box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1])
            ans.extend(coordinates)
    ans = ','.join([str(x) for x in ans]) 
    return ans


if __name__ == '__main__':
    application.run(host="0.0.0.0", port=80, debug=False)
