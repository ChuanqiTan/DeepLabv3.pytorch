"""
Usage:
    python ourdata_convert.py PARTENT/anno_v2
"""

import glob, os, sys, cv2
import numpy as np


def get_mask_png(img, masks):
    shape = img.shape
    out = np.zeros(shape=shape[:2])

    for m in masks:
        t, _, _, *polygon = m.strip().split(",")
        t = int(t)
        if 0 < t <= 10:
            p = []
            for i in range(0, len(polygon), 2):
                p.append([polygon[i], polygon[i+1]])

            p = np.array(p, np.int32)

            cv2.fillPoly(out,[p],(t,20*t,255))
            cv2.polylines(out,[p],True,(255,255,255), 2)

    return out


def main():
    jpgs = glob.glob("{}/*.jpg".format(sys.argv[1]))
    print("total has {} images".format(len(jpgs)))

    for idx, jpg in enumerate(jpgs[:]):
        print("{}/{} ...".format(idx, len(jpgs)))
        img = cv2.imread(jpg)
        with open(jpg.replace(".jpg", ".txt"), "r") as f:
            mask = f.readlines()

        mask_png = get_mask_png(img, mask)
        cv2.imwrite(jpg.replace(".jpg", ".png"), mask_png)


main()
