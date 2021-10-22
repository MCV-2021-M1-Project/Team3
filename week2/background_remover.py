import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.histograms import histogram

#Techniques like Wathershed, Canny, etc... Are considered "illegal"

class Canvas(object):

    def __init__(self):

        self.input_image #input_image
        self.adjust_gamma #adjust_gamma
        self.make_bin_and_objects #make_bin_and_abjects
        self.background_remover #backgound_remover
        self.connected_componets #connected_components
        self.simplify_irrelevant #simplyfy_irrelevant
        self.gray2rgb #gray2rgb
        self.save_mask #save_mask
        self.crop_image #crop_image

    def input_image(self,path):
        img = cv.imread(path)
        return img

    #----------------------------------------------------------

    def adjust_gamma(self,img, gamma=1.0):
        #----------- build a lookup table mapping the pixel values [0, 255] to
        #----------- their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        #----------- apply gamma correction using the lookup table
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

        max_h_value = max(h_tot)
        max_h_index = h_tot.index(max_h_value)+1
        ##print('Pos area max: ',max_h_index)
        backtorgb_mid = cv.cvtColor(nogaps,cv.COLOR_GRAY2RGB)
        ##print('Numbers of Components: ',numLabels-1)
        for f in range(1,numLabels):
            
            x = stats[f][0]
            y = stats[f][1]
            w = stats[f][2]
            h = stats[f][3]
            cv.rectangle(backtorgb_mid, (x, y), (x + w, y + h), (0, 255, 0), 10)
            ##print(stats[f])

        for f in range(1,numLabels):
            if f == max_h_index:
                continue
            
            x = stats[f][0]
            y = stats[f][1]
            w = stats[f][2]
            h = stats[f][3]
            cv.rectangle(nogaps, (x, y), (x + w, y + h), (0, 0, 0), -1)

        cx = int(centroids[max_h_index][0])
        cy = int(centroids[max_h_index][1])
        x = stats[max_h_index][0]
        y = stats[max_h_index][1]
        w = stats[max_h_index][2]
        h = stats[max_h_index][3]
        ##print('Centroids: X:',cx,'Y:',cy)
        return nogaps,cx,cy,x,y,w,h

    def connected_componets2(self,nogaps):
        #------------ Calculate connected components and delete the small components
        x2 = -1
        y2 = -1
        w2 = -1
        h2 = -1
        output = cv.connectedComponentsWithStats(
            nogaps, 4, cv.CV_32S)
        (numLabels, labels, stats, centroids) = output
        h_tot = []
        for f in range(1,numLabels):
            h_tot.append(stats[f][4])
            ##print(stats[f][4])
            ##print(h_tot)
        max_h_value = max(h_tot)
        #print('Max value1:',max_h_value)
        max_h_index = h_tot.index(max_h_value)+1
        #print('Max index1:',max_h_index)
        #print('Len_h_tot: ',len(h_tot))
        if len(h_tot) > 1:
            h_tot2 = h_tot
            #print(h_tot)
            #print(h_tot2)
            h_tot2[max_h_index-1] = 0
            #print('h_tot2: ',h_tot2)
            max_h_value2 = max(h_tot2)
            #print('Max value2:',max_h_value2)
            max_h_index2 = h_tot2.index(max_h_value2)+1
            #print('Max index2:',max_h_index2)

        #print('Pos area max: ',max_h_index)
        backtorgb_mid = cv.cvtColor(nogaps,cv.COLOR_GRAY2RGB)
        #print('Numbers of Components: ',numLabels-1)
        for f in range(1,numLabels):
            
            x = stats[f][0]
            y = stats[f][1]
            w = stats[f][2]
            h = stats[f][3]
            cv.rectangle(backtorgb_mid, (x, y), (x + w, y + h), (0, 255, 0), 10)
            ##print(stats[f])

        ##print('MAX INDEXES 1:',max_h_index,'MAX INDEXES 2:',max_h_index2)

        for f in range(1,numLabels):
            #print(f)
            x = stats[f][0]
            y = stats[f][1]
            w = stats[f][2]
            h = stats[f][3]
            cv.rectangle(nogaps, (x, y), (x + w, y + h), (0, 0, 0), -1)

        for f in range(1,numLabels):

            if f == (max_h_index):
                x = stats[f][0]
                y = stats[f][1]
                w = stats[f][2]
                h = stats[f][3]
                #------------------------------ White
                cv.rectangle(nogaps, (x, y), (x + w, y + h), (255, 255, 255), -1)
                #------------------------------ White
                #print(f,'SALTA')
                continue
            
            if max_h_index2 is not None:
                if f == (max_h_index2):
                    x2 = stats[f][0]
                    y2 = stats[f][1]
                    w2 = stats[f][2]
                    h2 = stats[f][3]
                    #print('DADWSDAW',x2,y2)
                    #------------------------------ White
                    cv.rectangle(nogaps, (x2, y2), (x2 + w2, y2 + h2), (255, 255, 255), -1)
                    #------------------------------ White
                    #print(f,'SALTA')
                    continue

        cx = int(centroids[max_h_index][0])
        cy = int(centroids[max_h_index][1])
        #print('Centroids: X:',cx,'Y:',cy)
        #plt.imshow(cv.cvtColor(backtorgb_mid, cv.COLOR_BGR2RGB))
        return nogaps,cx,cy,x,y,w,h,x2,y2,w2,h2

    def gray2rgb(self,nogaps):
        #------------ Convert the image
        backtorgb = cv.cvtColor(nogaps,cv.COLOR_GRAY2RGB)
        #cv.rectangle(backtorgb, (x, y), (x + w, y + h), (0, 255, 0), 3)
        #plt.imshow(cv.cvtColor(backtorgb, cv.COLOR_BGR2RGB))
        return backtorgb

    def save_mask(self,backtorgb,mask,img,cx,cy,save_path,f):
        #------------ Save the mask and the image + mask
        directory = save_path
        os.chdir(directory)
        filename = 'mask_'+f+'.png'
        cv.imwrite(filename, backtorgb)
        print('Successfully generated and saved',filename)

    def crop_image(self,img,save_directory_croped,x,y,w,h,x2,y2,w2,h2,f):
        file_name = os.path.splitext(f)[0]
        #print('FILE NAME:',file_name)
        croped_img = img[y:y+h, x:x+w]
        directory = save_directory_croped
        os.chdir(directory)
        filename = 'crop_'+file_name+'.jpg'
        cv.imwrite(filename, croped_img)
        print('Successfully generated and saved',filename)
        if x2 > 0:
            croped_img = img[y2:y2+h2, x2:x2+w2]
            directory = save_directory_croped
            os.chdir(directory)
            filename = 'crop_'+file_name+'_2'+'.jpg'
            cv.imwrite(filename, croped_img)
            #print('Successfully generated and saved',filename)


    def background_remover(self,path,save_path,save_path_croped,f):
        img = self.input_image(path)
        simplifyed_img, histogram, result = self.simplify_irrelevant(img)
        objects_img, mask = self.make_bin_and_objects(simplifyed_img)
        connected_img,cx,cy,x,y,w,h,x2,y2,w2,h2 = self.connected_componets2(objects_img)
        final_mask = self.gray2rgb(connected_img)
        self.save_mask(final_mask,mask,img,cx,cy,save_path,f)
        self.crop_image(img,save_path_croped,x,y,w,h,x2,y2,w2,h2,f)
        return final_mask,x,y,w,h,x2,y2,w2,h2

valid_images = [".jpg"]
load_directory = 'C:\\Users\\JQ\\Documents\\GitHub\\ABC\\CV_M1\\W2\\QSD2\\'
save_direcory = 'C:\\Users\\JQ\\Documents\\GitHub\\ABC\\CV_M1\\W2\\QSD2\\generated_masks'
save_directory_croped = 'C:\\Users\\JQ\\Documents\\GitHub\\ABC\\CV_M1\\W2\\QSD2\\croped'

if __name__ == "__main__":

    museum = Canvas()
    for f in os.listdir(load_directory):
        ##print(f)
        file_name = typex = os.path.splitext(f)[0]
        typex = os.path.splitext(f)[1]
        ##print(typex)
        if typex.lower() not in valid_images:
            continue
        museum.background_remover(load_directory + f,save_direcory,save_directory_croped ,file_name)
