from typing import final
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.histograms import histogram

#Techniques like Wathershed, Canny, etc... Are considered "illegal"

class Canvas(object):

    def __init__(self): #input_image, adjust_gamma, make_bin_and_abjects, backgound_remover,
    #connected_components, simplyfy_irrelevant, gray2rgb, save_mask

        self.input_image #input_image
        self.adjust_gamma #adjust_gamma
        self.make_bin_and_objects #make_bin_and_abjects
        self.background_remover #backgound_remover
        self.connected_componets #connected_components
        self.simplify_irrelevant #simplyfy_irrelevant
        self.gray2rgb #gray2rgb
        self.save_mask #save_mask

    def input_image(self,path):
        img = cv.imread(path)
        return img

    #----------------------------------------------------------

    def adjust_gamma(self,img, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv.LUT(img, table)

    #----------------------------------------------------------

    def make_bin_and_objects(self,blurred):
        #------------ Make binarisation
        (T, threshInv) = cv.threshold(blurred, 0, 255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        #------------ Erode and dilate image to remove details
        eroded = cv.erode(threshInv,np.ones((25,25)))
        eroded = cv.dilate(eroded,np.ones((25,25)))
        eroded = cv.erode(threshInv,np.ones((25,25)))
        eroded = cv.dilate(eroded,np.ones((25,25)))
        #------------ Fill the remaining backgorund
        h, w = eroded.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        eroded_copy = eroded.copy()
        cv.floodFill(eroded_copy, mask, (1,1), (0,0,0))
        #------------ invert de image and join them
        eroded_inv = cv.bitwise_not(eroded)
        nogaps = eroded_copy | eroded_inv
        return nogaps,mask

    #----------------------------------------------------------

    def simplify_irrelevant(self,img):
        rgb_planes = cv.split(img)
        result_planes = []
        result_norm_planes = []

        for plane in rgb_planes:
            dilated_img = cv.dilate(plane, np.ones((7,7), np.uint8))
            bg_img = cv.medianBlur(dilated_img, 21)
            diff_img = 255 - cv.absdiff(plane, bg_img)
            norm_img = cv.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)

        result = cv.merge(result_planes)
        result_norm = cv.merge(result_norm_planes)

        plt.imshow(cv.cvtColor(result_norm, cv.COLOR_BGR2RGB))
        gray = result_norm

        #------------ calculate histogram
        gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (7, 7), 0)
        histogram = cv.calcHist([blurred], [0], None, [256], [0, 256])
        return blurred, histogram, result
        
    def connected_componets(self,nogaps):
        #------------ Calculate connected components and delete the small components
        output = cv.connectedComponentsWithStats(
            nogaps, 4, cv.CV_32S)
        (numLabels, labels, stats, centroids) = output
        h_tot = []
        for f in range(1,numLabels):
            h_tot.append(stats[f][4])
            #print(stats[f][4])
            #print(h_tot)
        max_h_value = max(h_tot)
        max_h_index = h_tot.index(max_h_value)+1
        print('Pos area max: ',max_h_index)
        backtorgb_mid = cv.cvtColor(nogaps,cv.COLOR_GRAY2RGB)
        print('Numbers of Components: ',numLabels-1)
        for f in range(1,numLabels):
            x = stats[f][0]
            y = stats[f][1]
            w = stats[f][2]
            h = stats[f][3]
            cv.rectangle(backtorgb_mid, (x, y), (x + w, y + h), (0, 255, 0), 10)
            #print(stats[f])

        for f in range(1,numLabels):
            if f == max_h_index:
                continue
            
            x = stats[f][0]
            y = stats[f][1]
            w = stats[f][2]
            h = stats[f][3]
            cv.rectangle(nogaps, (x, y), (x + w, y + h), (0, 0, 0), -1)
            #print(stats[f])
        cx = int(centroids[max_h_index][0])
        cy = int(centroids[max_h_index][1])
        print('Centroids: X:',cx,'Y:',cy)
        plt.imshow(cv.cvtColor(backtorgb_mid, cv.COLOR_BGR2RGB))
        return nogaps,cx,cy

    def gray2rgb(self,nogaps):
        #------------ Convert the image
        backtorgb = cv.cvtColor(nogaps,cv.COLOR_GRAY2RGB)
        #cv.rectangle(backtorgb, (x, y), (x + w, y + h), (0, 255, 0), 3)
        plt.imshow(cv.cvtColor(backtorgb, cv.COLOR_BGR2RGB))
        return backtorgb

    def save_mask(self,backtorgb,mask,img,cx,cy,save_path,f):
        #------------ Save the mask and the image + mask
        directory = save_path
        os.chdir(directory)
        print("Before saving image:")  
        print(os.listdir(directory))  
        filename = 'mask_'+f+'.png'
        cv.imwrite(filename, backtorgb)
        print("After saving image:")  
        print(os.listdir(directory))
        print('Successfully saved')

        #backtorgb = cv.imread('C:\\Users\\usuario\\Documents\\GitHub\\ABC\\CV_M1\\W1\\mask.png') #, cv.IMREAD_GRAYSCALE
        #cv.floodFill(backtorgb, mask, (cx,cy), (0,255,0))
        #added_image = cv.addWeighted(img,0.5,backtorgb,0.5,0)

        #plt.imshow(cv.cvtColor(added_image, cv.COLOR_BGR2RGB))

    def background_remover(self,path,save_path,f):
        img = self.input_image(path)
        #cv.imshow('',img)
        #cv.waitKey(0)
        simplifyed_img, histogram, result = self.simplify_irrelevant(img)
        #cv.imshow('',simplifyed_img)
        #cv.waitKey(0)
        objects_img, mask = self.make_bin_and_objects(simplifyed_img)
        connected_img,cx,cy = self.connected_componets(objects_img)
        print('awd:',cx)
        final_mask = self.gray2rgb(connected_img)
        self.save_mask(final_mask,mask,img,cx,cy,save_path,f)
        #cv.imshow('',final_mask)
        #cv.waitKey(0)

valid_images = [".jpg"]
load_directory = 'C:\\Users\\usuario\\Documents\\GitHub\\ABC\\CV_M1\\W1\\QSD2\\'
save_direcory = 'C:\\Users\\usuario\\Documents\\GitHub\\ABC\\CV_M1\\W1\\QSD2\\generated_masks'
if __name__ == "__main__":

    museum = Canvas()
    for f in os.listdir('C:\\Users\\usuario\\Documents\\GitHub\\ABC\\CV_M1\\W1\\QSD2'):
        print(f)
        file_name = typex = os.path.splitext(f)[0]
        typex = os.path.splitext(f)[1]
        print(typex)
        if typex.lower() not in valid_images:
            continue
        museum.background_remover(load_directory + f,save_direcory,file_name)

