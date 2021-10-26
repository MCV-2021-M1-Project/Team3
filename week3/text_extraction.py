import numpy as np
import cv2 
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
class Text(object):

    def __init__(self):

        self.input_image #input_image
        self.gray2rgb #adjust_gamma
        self.pre_process #make_bin_and_abjects
        self.search_elements #backgound_remover
        self.generate_mask #connected_components
        self.text_extraction #simplyfy_irrelevant
        self.gray2rgb #gray2rgb
        self.save_mask #save_mask
        #self.crop_image #crop_image

    def input_image(self,path):
        if type(path) is str:
            img = cv2.imread(path)
        else:
            img = path
        return img

    def gray2rgb(self,nogaps):
        #------------ Convert the image
        backtorgb = cv2.cvtColor(nogaps,cv2.COLOR_GRAY2RGB)
        #cv.rectangle(backtorgb, (x, y), (x + w, y + h), (0, 255, 0), 3)
        #plt.imshow(cv2.cvtColor(backtorgb, cv2.COLOR_BGR2RGB))
        return backtorgb

    def pre_process(self,img):

        kernel = np.ones((30, 30), np.uint8) 
        img_TH = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        img_BH = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

        TH = 150
        img_TH[(img_TH[:,:,0] < TH) | (img_TH[:,:,1] < TH) | (img_TH[:,:,2] < TH)] = (0,0,0)
        img_BH[(img_BH[:,:,0] < TH) | (img_BH[:,:,1] < TH) | (img_BH[:,:,2] < TH)] = (0,0,0)

        kernelzp = np.ones((1, int(img.shape[1] / 8)), np.uint8) 
        img_TH = cv2.dilate(img_TH, kernelzp, iterations=1) 
        img_TH = cv2.erode(img_TH, kernelzp, iterations=1) 
        img_BH = cv2.dilate(img_TH, kernelzp, iterations=1) 
        img_BH = cv2.erode(img_TH, kernelzp, iterations=1) 

        img_sum = img_TH + img_BH

        return (cv2.cvtColor(img_sum, cv2.COLOR_BGR2GRAY) != 0).astype(np.uint8)

    def search_elements(self,img,threshInv):
        output = cv2.connectedComponentsWithStats(
            threshInv, 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        h_tot = []
        for f in range(1,numLabels):
            h_tot.append(stats[f][4])
            ##print(stats[f][4])
            ##print(h_tot)

        w,h = img.shape[0],img.shape[1]
        areatg = w*h
        #max_h_value = max(h_tot)
        max_h_value = 0
        max_h_index = 0
        ##print(max_h_value)
        #print('stats[0][4]',stats[0][4])
        #max_h_index = h_tot.index(max_h_value)+1

        for h in range(0,numLabels-1):
            area1 = stats[h+1][2]*stats[h+1][3]
            if (area1)>(areatg/2):
                continue
            if max_h_value < h_tot[h]:
                max_h_value = h_tot[h]
                #print('max:value',max_h_value)
        if h_tot:
            max_h_index = h_tot.index(max_h_value)+1

        #print(max_h_value)
        #print('Pos area max: ',max_h_index)
        backtorgb_mid = cv2.cvtColor(threshInv,cv2.COLOR_GRAY2RGB)
        #print('Numbers of Components: ',numLabels-1)
        for f in range(1,numLabels):
            
            x = stats[f][0]
            y = stats[f][1]
            w = stats[f][2]
            h = stats[f][3]
            cv2.rectangle(backtorgb_mid, (x, y), (x + w, y + h), (0, 255, 0), 10)
            ##print(stats[f])

        for f in range(1,numLabels):
            if f == max_h_index:
                continue
            
            x = stats[f][0]
            y = stats[f][1]
            w = stats[f][2]
            h = stats[f][3]
            cv2.rectangle(threshInv, (x, y), (x + w, y + h), (0, 0, 0), -1)
            ##print(stats[f])
        #print(stats[max_h_index])
        x1 = stats[max_h_index][0]
        y1 = stats[max_h_index][1]
        w1 = stats[max_h_index][2]
        h1 = stats[max_h_index][3]
        bbox = [x1,y1,w1,h1]
        #print(bbox)
        cx = int(centroids[max_h_index][0])
        cy = int(centroids[max_h_index][1])
        #print('Centroids: X:',cx,'Y:',cy)
        #plt.imshow(cv2.cvtColor(backtorgb_mid, cv2.COLOR_BGR2RGB))
        #plt.show()
        return bbox

    def generate_mask(self,img,bbox):
        x1 = bbox[0]
        y1 = bbox[1]
        w1 = bbox[2]
        h1 = bbox[3]
        polygon = [(x1,y1),(x1+w1,y1+h1)]
        w,h = img.shape[0],img.shape[1]
        img = np.zeros((w,h,3),dtype=np.uint8)
        mask = cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(255,255,255),-1)
        return mask

    def save_mask(self,backtorgb,save_path,f):
        #------------ Save the mask and the image + mask
        filename = 'mask_' + f + '.png'
        cv2.imwrite(os.path.join(save_path, filename), backtorgb)
        print('Successfully generated and saved',filename)
        #plt.imshow(backtorgb)
        #plt.show()

    def text_extraction(self,path,save_path,f):
        img = self.input_image(path)
        sum2 = self.pre_process(img)
        (T, threshInv) = cv2.threshold(sum2, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        bbox = self.search_elements(img,threshInv)
        backtorgb = self.gray2rgb(threshInv)
        mask = self.generate_mask(img,bbox)
        backtorgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        if save_path is not None:
            self.save_mask(backtorgb,save_path,f)

        bbox[2],bbox[3] = bbox[0]+bbox[2],bbox[1]+bbox[3]
        print('BBOX: ',bbox)
        print('---------------------------------------------------------')
        return bbox,mask

valid_images = [".jpg"]
path = 'C:\\Users\\JQ\\Documents\\GitHub\\ABC\\CV_M1\\W2\\QSD2\\croped'
save_path = 'C:\\Users\\JQ\\Documents\\GitHub\\ABC\\CV_M1\\W2\\QSD2\\generated_text_masks'
path = "datasets/qsd1_w2/"
save_path = "datasets/qsd1_w2/generated_text_masks"
if __name__ == "__main__":

    text_id = Text()
    for f in os.listdir(path):
        ##print(f)
        file_name = typex = os.path.splitext(f)[0]
        typex = os.path.splitext(f)[1]
        ##print(typex)
        if typex.lower() not in valid_images:
            continue
        #path_in = path + '\\' + f
        path_in = os.path.join(path, f)
        #print(path_in)
        text_id.text_extraction(path_in,save_path,file_name)
