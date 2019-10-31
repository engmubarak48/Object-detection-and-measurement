# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:42:45 2019

@author: Jama Hussein Mohamud
"""

import cv2
import json
import numpy as np
from shapely import affinity 
from shapely.geometry import Polygon

#%%
## Function to show the image
def show_image(image):
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def show_croped(image):
#    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#%%
# Removing small Objects
def remove_small(image, data, efact, xfact, yfact):
    for polygon in np.arange(len(data['shapes'])):
        data['shapes'][polygon]['label'] = data['shapes'][polygon]['label'].lower()
        if data['shapes'][polygon]['label'] == 'a':
             pts = data['shapes'][polygon]['points']
             x = []
             y = []
             for i in pts:
                 x.append(i[0])
                 y.append(i[1])
             minx = min(x)
             maxx = max(x)
             miny = min(y)
             maxy = max(y)
             pts[0][0], pts[1][0], pts[2][0], pts[3][0] = minx, minx, maxx, maxx
             pts[0][1], pts[1][1], pts[2][1], pts[3][1] = miny, maxy, maxy , miny
             pts = [[pts[0][0]-efact, pts[0][1]-efact], [pts[1][0]-efact, pts[1][1]+efact], [pts[2][0]+efact, pts[2][1]+efact], [pts[3][0]+efact, pts[3][1]-efact]]
             pts = np.array(pts)
             x,y,w,h = cv2.boundingRect(pts.astype(int))
             image = cv2.rectangle(image.copy(), (x,y),(x+w,y+h),(255,255,255), -1)
#             image = cv2.fillPoly(image.copy(), pts =[pts], color=(255,255,255))
        if (data['shapes'][polygon]['label'] == 'k' or data['shapes'][polygon]['label'] == 'c'):
            pts = data['shapes'][polygon]['points']
            pts = [tuple(x) for x in pts]
            scaled = affinity.scale(Polygon(pts), xfact=xfact, yfact=yfact, origin='center')
            pts = scaled.exterior.coords[:]
            pts = [[int(j) for j in i] for i in pts]
            pts = np.array(pts)
            image = cv2.fillPoly(image.copy(), pts =[pts.astype(int)], color=(255,255,255))
        if data['shapes'][polygon]['label'] == 'u':
            pts = data['shapes'][polygon]['points']
            pts = np.array(pts)
            image = cv2.fillPoly(image.copy(), pts =[pts.astype(int)], color=(255,255,255))
    return image
#%%
# Getting our template
def roi(image, data):
    for polygon in np.arange(len(data['shapes'])):
        data['shapes'][polygon]['label'] = data['shapes'][polygon]['label'].lower()
        if data['shapes'][polygon]['label'] == 'b':
            pts = np.array(data['shapes'][polygon]['points'])
            mask = np.zeros(image.shape[:2], np.uint8)
            cv2.drawContours(mask, [pts.astype(int)], -1, (255, 255, 255), -1, cv2.LINE_AA)
            dst = cv2.bitwise_and(image, image, mask=mask)
            ## (4) add the white background
            bg = np.ones_like(image, np.uint8)*255
            cv2.bitwise_not(bg,bg, mask=mask)
            dst2 = bg+ dst
    return dst2
#%%
# function to check if there are extra objects        
def check_extra_object(image, data, efact, xfact, yfact):
    image = cv2.GaussianBlur(image, (5,5),0)
#    show_image(image)
    image = roi(image, data)
#    show_image(image)
    image = remove_small(image, data, efact, xfact, yfact)
#    show_image(image)
    ret, thresh = cv2.threshold(image.copy(), 130, 255, cv2.THRESH_BINARY_INV)
#    thresh = cv2.adaptiveThreshold(image.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
#    show_image(thresh)
    closing = cv2.morphologyEx(thresh.copy(), cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    erosion = cv2.erode(closing.copy(), np.ones((5,5),np.uint8), iterations = 1)
    _, contours, _ = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
#    show_image(erosion)
    return erosion, contours
#%%
def polygon_anlysis(image, mask, data, polygon, pfact, oxfact, oyfact):
    pts1 = data['shapes'][polygon]['points']
    pts1 = np.array(pts1)
    x,y,w,h = cv2.boundingRect(pts1.astype(int))
    if data['shapes'][polygon]['label'] == 'a':
        pts = data['shapes'][polygon]['points']
        X = []
        Y = []
        for i in pts:
            X.append(i[0])
            Y.append(i[1])
        minx, maxx, miny, maxy = min(X), max(X), min(Y), max(Y)
        pts[0][0], pts[1][0], pts[2][0], pts[3][0] = minx, minx, maxx, maxx
        pts[0][1], pts[1][1], pts[2][1], pts[3][1] = miny, maxy, maxy , miny
        pts = [[pts[0][0]-pfact, pts[0][1]-pfact], [pts[1][0]-pfact, pts[1][1]+pfact], [pts[2][0]+pfact, pts[2][1]+pfact], [pts[3][0]+pfact, pts[3][1]-pfact]]
        pts = np.array(pts)
        x1,y1,w1,h1 = cv2.boundingRect(pts.astype(int))
        croped = image[y1:y1+h1, x1:x1+w1].copy()
#        show_croped(croped)
        ret, thresh = cv2.threshold(croped, 169, 255, cv2.THRESH_BINARY)
#        if polygon == 2:
#            show_croped(croped)
#            show_croped(thresh)
    elif data['shapes'][polygon]['label'] == 'k':
        pts = data['shapes'][polygon]['points']
        pts = [tuple(x) for x in pts]
        scaled = affinity.scale(Polygon(pts), oxfact, oyfact, origin='center')
        pts = scaled.exterior.coords[:]
        pts = [[int(j) for j in i] for i in pts]
        pts = np.array(pts)
        cv2.fillPoly(mask, [pts.astype(int)], (255, 255, 255))
#         apply the mask
        masked_image = cv2.bitwise_and(image, mask)
        croped = masked_image[y:y+h, x:x+w].copy()
#        show_croped(croped)
        ret, thresh = cv2.threshold(croped, 190, 255, cv2.THRESH_BINARY)
#        ret, thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#        show_croped(thresh) 
    elif data['shapes'][polygon]['label'] == 'c':
        pts = data['shapes'][polygon]['points']
        pts = [tuple(x) for x in pts]
        scaled = affinity.scale(Polygon(pts), xfact=1, yfact=1, origin='center')
        pts = scaled.exterior.coords[:]
        pts = [[int(j) for j in i] for i in pts]
        pts = np.array(pts)
        cv2.fillPoly(mask, [pts.astype(int)], (255, 255, 255))
#         apply the mask
        masked_image = cv2.bitwise_and(image, mask)
        croped = masked_image[y:y+h, x:x+w].copy()
#        show_croped(croped)
        ret, thresh = cv2.threshold(croped, 183, 255, cv2.THRESH_BINARY)
    else:
        croped = image[y:y+h, x:x+w].copy()
        ret, thresh = cv2.threshold(croped, 150, 255, cv2.THRESH_BINARY)
    #get percentage of pixels of each color.   
    Total_pixels = thresh.shape[0] * thresh.shape[1]
    white_pixels = cv2.countNonZero(thresh) # Nonzero pixels
    black_bixels = Total_pixels - white_pixels 
    # Get percentage of white pixels
    per_white_pixels =  white_pixels/Total_pixels * 100
    #get percentage of black pixels
    per_black_pixels =  black_bixels/Total_pixels * 100
    return pts1, per_white_pixels, per_black_pixels
#%%
# function to check if the object is located at a wrong location
def check_object(image, data, MinArea, thresholdm, maxArea, thresholde, thresholda, thresholdc, thresholdk, pfact, oxfact, oyfact):
    dic1 = {}
    dic2 = {}
#    image = cv2.GaussianBlur(image,(5,5),0)
#    image = cv2.fastNlMeansDenoising(image, None,10,7,21)
    image = cv2.medianBlur(image, 5)
    mask = np.zeros(image.shape, dtype=np.uint8)
    for polygon in np.arange(len(data['shapes'])):
        data['shapes'][polygon]['label'] = data['shapes'][polygon]['label'].lower()
        if data['shapes'][polygon]['label'] == 'a': # To not include the template and unneeded objects
            pts1, per_white_pixels, per_black_pixels = polygon_anlysis(image, mask, data, polygon, pfact, oxfact, oyfact)
            pts = data['shapes'][polygon]['points']
#            print(np.array_equal(np.array(pts), pts1))
            pts = [tuple(x) for x in pts]
            Area = Polygon(pts).area
            if Area > maxArea and per_white_pixels > thresholda:
                dic1 = {**dic1, str(polygon): [pts1, Area, per_white_pixels, per_black_pixels]}
            elif MinArea <= Area <= maxArea and per_white_pixels > thresholde:
                dic1 = {**dic1, str(polygon): [pts1, Area, per_white_pixels, per_black_pixels]}
            elif Area < MinArea and per_white_pixels > thresholdm:
                dic1 = {**dic1, str(polygon): [pts1, Area, per_white_pixels, per_black_pixels]}
            else:
                dic2 = {**dic2, str(polygon): [pts1, Area, per_white_pixels, per_black_pixels]}
        elif data['shapes'][polygon]['label'] == 'c': 
            pts1, per_white_pixels, per_black_pixels = polygon_anlysis(image, mask, data, polygon, pfact, oxfact, oyfact)
            if per_white_pixels > thresholdc:
                dic1 = {**dic1, str(polygon): [pts1, 1, per_white_pixels, per_black_pixels]}
            else:
                dic2 = {**dic2, str(polygon): [pts1, 1, per_white_pixels, per_black_pixels]}
        elif data['shapes'][polygon]['label'] == 'k': 
            pts1, per_white_pixels, per_black_pixels = polygon_anlysis(image, mask, data, polygon, pfact, oxfact, oyfact)
            if per_white_pixels > thresholdk:
                dic1 = {**dic1, str(polygon): [pts1, 1, per_white_pixels, per_black_pixels]}
            else:
                dic2 = {**dic2, str(polygon): [pts1, 1, per_white_pixels, per_black_pixels]}
    return dic1, dic2

#%%
def draw_extra_objects(image, contours, dic1):
    for cnt in contours:
        cv2.drawContours(image, [cnt], 0, (0,255,0), 3)
    draw_wrong_objects(image, dic1)
    return image
#%%
def draw_wrong_objects(image, dic1):
    for key in dic1:
        pts = dic1[str(key)][0]
        x,y,w,h = cv2.boundingRect(pts.astype(int))
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 3)
    return image
#%%
# Check the erkurt
def check_erkurt(image, data, pfact, efact, xfact, yfact, oxfact, oyfact, MinArea, thresholdm, maxArea, thresholde, thresholda, thresholdc, thresholdk, Area, RecP, width, height):
    img, contours = check_extra_object(image.copy(), data, efact, xfact, yfact)
    cnts = []
    for cnt in contours:
        Rarea = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.04*cv2.arcLength(cnt,True),True)
        x, y, w, h = cv2.boundingRect(cnt)
        if len(approx) >= RecP and Rarea > Area and w > width and h > height:
            cnts.append(cnt)
    dic1, dic2 = check_object(image.copy(), data, MinArea, thresholdm, maxArea, thresholde, thresholda, thresholdc, thresholdk, pfact, oxfact, oyfact)
    image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    if len(dic1) > 0 and len(cnts) > 0:
        text = "NOK- There are - {0} - extra objects- And, {1} objects are at wrong location".format(len(cnts), len(dic1))
        fimage = draw_extra_objects(image.copy(), cnts, dic1)
    elif len(dic1) > 0 and len(cnts) == 0:
        text = "NOK- {} objects are located at wrong location-".format(len(dic1))
        fimage = draw_wrong_objects(image.copy(), dic1)
    elif len(dic1) == 0 and len(cnts) > 0:
        text = "NOK- There are - {0} - extra objects".format(len(cnts))
        fimage = draw_extra_objects(image.copy(), cnts, dic1)
    else:
        fimage = image.copy()
        text = "OKEY"
    return fimage, dic1, dic2, img, text, cnts
#%%
with open("D:/Anadolu University/My Major/Assproject/Measurement and alignment/Erkurt/calisma/10062019/06-aag/calibration/6.json") as f:
    data = json.load(f)

# load the input image from disk
registered_image = cv2.imread("D:/Anadolu University/My Major/Assproject/Measurement and alignment/Erkurt/calisma/10062019/06-aag/calibration/10.png", 0)

#%%
#(image, data, pfact, efact, xfact, yfact, oxfact, oyfact, MinArea, thresholdm, maxArea, thresholde, thresholda, thresholdc, thresholdk, Area, RecP, width, height)

fimage, dic1, dic2, img, text, cnts = check_erkurt(registered_image, data, 0, 1.01, 1, 1, 1, 1, 900, 20, 1500, 18, 15, 20, 15, 200, 2, 10, 10)
show_image(fimage)












