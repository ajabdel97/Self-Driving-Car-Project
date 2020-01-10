import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
%matplotlib inline

import os
#Importing Images (Dataset):
def readImages(dir, pattern):

    images = []
    for dirpath, dirnames, filenames in os.walk(dir):
        for dirname in dirnames:
            images.append(glob.glob(dir +'/' + dirname + '/' + pattern))

    flatten = [item for sublist in images for item in sublist]
    return list(map(lambda img: cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB), flatten))

vehicles = readImages('./training_images/vehicles', '*.png')
non_vehicles = readImages('./training_images/non-vehicles', '*.png')
ncols = 2
index = 10

vehicle = vehicle[index]
non_vehicle = non_vehicle[index]

fig, axes = plt.subplots(ncols, figsize=(10, 10))

axes[0].imshow(vehicle)
axes[0].set_title('vehicle')
axes[1].imshow(non_vehicle)
axes[1].set_title('Non vehicle')

print('Vehicle train image count: {}'.format(len(vehicles)))
print('Non- Vehicle train image count: {}'.format(len(non_vehicles)))

#Feature extraction:

def bin_spatial(img, size=(32, 32)):

    features = cv2.resize(img, size).ravel()
    return features

def color_hist(img, nbins=32, bins_range=(0, 256)):

    channel1_hist = np.histogram(img[:,:,0], bins= nbins, range= bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins= nbins, range= bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins= nbins, range= bins_range)

    hist_features = np.concatenate((channel1_hist[0],channel2_hist[0], channel3_hist[0]))
    return hist_features

from skimage.feature import hog

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis = False, feature_vec= True):

    if vis == True:
        features, hog_image = hog(img, orientations= orient, pixels_per_cell=(pix_per_cell,pix_per_cell),
                                  cell_per_block=(cell_per_block, cell_per_block), transform_sqrt= True,
                                  visualise= vis, feature_vector = feature_vec)
        return features , hog_image

    else:
        features, hog_image = hog(img, orientations= orient, pixels_per_cell=(pix_per_cell,pix_per_cell),
                                  cell_per_block=(cell_per_block, cell_per_block), transform_sqrt= True,
                                  visualise= vis, feature_vector = feature_vec)
        return features

class FeaturesParameters():
    def __init__(self):

        self.cspace = 'YCrCb'
        self.orient = 8
        self.pix_per_cell = 8
        self.cell_per_block = 2
        self.hog_channel = 'ALL'
        self.size = (16, 16)
        self.his_bins = 32
        self.hist_range = (0, 256)


def extract_features(image, params):

    cspace = params.cspace
    orient = params.orient
    pix_per_cell = params.cell_per_block
    cell_per_block = params.cell_per_block
    hog_channel = params.hog_channel
    size = params.size
    hist_bins = params.hist_bins
    hist_range = params.hist_range

    if cspae != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)

    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel],
                                                 orient, pix_per_cell, cell_per_block,
                                                 vis = False, feature_vec= True))

        hog_features = np.ravel(hog_features)
    else:
        hog_features= get_hog_features(feature_image[:,:, hog_channel] ,orient,
                                        pix_per_cell, cell_per_block,vis = False, feature_vec= True))

    spatial_features = bin_spatial(feature_image, size)

    hist_features = color_hist(feature_image, nbins= hist_bins, bins_range= hist_range)

    return np.concatenate((spatial_features, hist_features, hog_features))

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
def fitModel(positive, negative, svc, scaler, params):

    positive_features = list(map(lambda img: extract_features(img, params, positive)))
    negative_features = list(map(lambda img: extract_features(img, params, negative)))

    X = np.vstack((positive_features, negative_features)).astype(np.float64)
    X_scaler = scaler.fit(X)
    scaled_X = X_scaler.transform(X)

    y = np.hstack((np.ones(len(positive_features)), np.zeros(len(negative_features))))
    rand_state = np.random.randint(0, 100)
    X_train , X_test, y_train, y_test = train_test_test_split(scaled_X, y, test_size = 0.2, random_state= rand_state)

    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()

    fittingTime = round (t2-t, 2)
    accuracy = round(svc.score(X_test, y_test), 4)
    return (svc, X_scaler, fittingTime, accuracy)

from sklearn.svm import LinearSVC

params = FeaturesParameters()
svc, scaler, fittingTime, acuracy = fitModel(vehicles, non_vehicles, LinearSVC(), StandardScaler(), params)
print('Fitting time: {} s, Accuracy: {}'.format(fittingTime, accuracy))


#sHOWING hOG IMAGES FROM training:

def showHOG(img, title):

    img_cspaced = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    _, hog_y = get_hog_features(img_cspaced[:,:,0],orient,pix_per_cell, cell_per_block,
                                vis = True, feature_vec= True))
    _, hog_Cr = get_hog_features(img_cspaced[:,:,1],orient,pix_per_cell, cell_per_block,
                                vis = True, feature_vec= True))
    _, hog_Cb = get_hog_features(img_cspaced[:,:,2],orient,pix_per_cell, cell_per_block,
                                vis = True, feature_vec= True))

    fig, axes = plt.subplots(ncols= 4, figsize= (15, 15))
    axes[0].imshow(img)
    axes[0].set_title(title)
    axes[1].imshow(hog_y, cmap='gray')
    axes[1].set_title('HOG -Y')
    axes[2].imshow(hog_Cr, cmap='gray')
    axes[2].set_title('HOG -Cr')
    axes[3].imshow(hog_Cb, cmap='gray')
    axes[3].set_title('HOG -Cb')

showHOG(vehicle, 'vehicle')
showHOG(vehicle, 'Non-vehicle')

#Slide window:

def draw_boxes(img, bboxes, color= (0, 0, 255), thick=6):

    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy

def slide_window(img, x_start_stop= [None, None], y_start_stop= [None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):

    if x_start_stop[0] == None:
       x_start_stop[0] = 0
    if x_start_stop[1] == None:
       x_start_stop[1] = img.shape[1]
    
    if y_start_stop[0] == None:
       y_start_stop[0] = 0
    if y_start_stop[1] == None:
       y_start_stop[1] = img.shape[0]

    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))

    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(wy_window[1]*(xy_overlap[1]))

    nx_windows = np.int((xspan - nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer)/ny_pix_per_step)
    window_list = []

    for ys in range(ny_windows):
        for xs in range(nx_windows):
            startx =  xs* nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys* ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            window_list.append((startx, starty), (endx, endy))

    return window_list

test_images = list(map(lambda img: cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB), glob.glob('./test_images/*.jpg')))

def findCarWindows(img, clf , scaler, params, y_start_stop= [360, 700], xy_window=(64, 64), xy_overlap=(0.85, 0.85)):

    car_windows = []
    windows = slide_window(img, x_start_stop = [None, None], y_start_stop0 xy_window= xy_window, xy_overlap)
    for window in windows:
        img_window = cv2.resize(img[window[0][1]:window[1][0],window[0][1]: window[1][1]], (64, 64))
        features = extract_features(img_window, params)
        scaled_features = scaler.transform(features.reshape(1, -1))
        pred = clf.predict(scaled_features)
        if pred == 1:
            car_windows.append(window)
    return car_windows

def drawCars(img, windows):

    output = np.copy(img)
    return draw_boxes(output, windows)

car_on_test = list(map(lambda img: drawCars(img, findCarWindows(img, svc, scaler, params)), test_images))

def showImages(images, cols = 2, rows =3, figsize=(15,13)):

    imgLength = len(images)
    fig, axes = pltsubplots(rows, cols, figsize= figsize)
    indexes = range(cols * rows)
    for ax, index in zip(axes.flat, indexes):
        if index < imgLength:
            image = images[index]
            ax.imshow(image)

showImages(car_on_test)


#Heat map and labels:

def add_heat(heatmap, bbox_list):

    for box in bbox_list:

        heatmap[box[0][1]: box[1][1], box[0][0]: box[1][0]] += 1
    return heatmap

def apply_threshold(heatmap, threshold):

    heatmap[heatmap <= threshold] = 0
    return heatmap

def draw_labeled_bboxes(img, labels):

    for car_number in range(1, labels[1]+1):

        nonzero =(labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),(np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    return img

from scipy.ndimage.measurements import label

def drawCarsWithLabels(img, boxes, threshHold = 4):

    heatmap = add_heat(np.zeros(img.shape), boxes)
    heatmap = apply_threshold(heatmap, threshHold)
    labels = label(heatmap)

    return draw_labeled_bboxes(np.copy(img), labels)

boxed_on_test = list(map(lambda img: drawCarsWithLabels(img, findCarWindows(img, svc, scaler, params)), test_images))

showImages(boxed_on_test)

#Improving performance with HOG sub sampling:

def findBoxes(img, clf, scaler, params, y_start_stop = [350, 656], window = 64, cell_per_step= 1, scale= 1.5):
    
    cspace = params.cspace
    orient = params.orient
    pix_per_cell = params.pix_per_cell
    cell_per_block = parmas.cell_per_block
    hog_channel = params.hog_channel
    size = params.size
    hist_bins = params.hist_bins
    hist_range = params.hist_range
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)

    ystart, ystop = y_start_stop
    ctrans_tosearch = feature_image[ystart:ystop,:,:]
    if scale != 1:
        imshape = ctrans.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale),np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    nxblocks = (ch1.shape[1]// pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0]// pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient* cell_per_block**2

    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    nxstep = (nxblocks - nblocks_per_window) // cells_per_step
    nystep = (nyblocks - nblocks_per_window) // cells_per_step

    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    car_windows = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            hog_feat1 = hog1[ypos:ypos+nblocks_per_window,xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window,xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window,xpos:xpos+nblocks_per_window].ravel()

            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos* pix_per_cell
            submig = cv2.resize(ctrans_tosearch[y_top: y_top+ window, xleft: xleft+ window], (64,64))
            spatial_features = bin_spatial(submig, size= size)
            hist_features = color_hist(submig, nbins = hist_bins, bins_range = hist_range)
            test_features = scaler.transform(np.hstack((spatial_features, hist_features, hog_features))).reshape(1, -1)
            test_predition = clf.predict(test_features)
            if test_preiction == 1:
                xbox_left = np.int(x_left*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                car_windows.append(((xbox_left, ytop_draw + ystart), (xbox_left+ win_draw, ytop_draw + win_draw+ ystart)))

            return car_windows

fast_boxes = list(map(lambda img: finBoxes(img, svc, scaler, params), test_images))

fast_on_test = list(map(lambda imgAndBox: drawCars(imgAndBox[0], imgAndBox[1]),zip(test_images, fast_on_test)))
showImages(fast_on_test)


fast_on_test = list(map(lamba imgAndBox: drawCarsWithLabels(imgAndBox[0], imgAndBox[1], threshold=4), zip(test_images, fast_boxes)))
showImages(fast_on_test)

                
                                            
