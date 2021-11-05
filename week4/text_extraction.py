import numpy as np
import cv2 
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import pytesseract
import string
import re
import textdistance
from math import exp

#pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\usuario\AppData\\Local\\Tesseract-OCR\\tesseract'

TEXT_SIMILARITIES_ALG = {
    'haming':textdistance.hamming,
    'mlipns':textdistance.mlipns,
    'ratcliff_obershelp': textdistance.ratcliff_obershelp,
    'gotoh':textdistance.gotoh,
    'cosine_similarity': textdistance.cosine,
    'levenshtein': textdistance.levenshtein.normalized_similarity,
    }

class Text(object):

    def __init__(self):

        self.input_image #input_image
        self.gray2rgb #adjust_gamma
        self.pre_process #make_bin_and_abjects
        self.search_elements #search_elements
        self.generate_mask #generate_mask
        self.text_extraction #text_extraction
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
        w,h = img.shape[0],img.shape[1]
        areatg = w*h

        max_h_value = 0
        max_h_index = 0

        for h in range(0,numLabels-1):
            area1 = stats[h+1][2]*stats[h+1][3]
            if (area1)>(areatg/2):
                continue
            if max_h_value < h_tot[h]:
                max_h_value = h_tot[h]
        if h_tot:
            max_h_index = h_tot.index(max_h_value)+1

        backtorgb_mid = cv2.cvtColor(threshInv,cv2.COLOR_GRAY2RGB)
        for f in range(1,numLabels):
            
            x = stats[f][0]
            y = stats[f][1]
            w = stats[f][2]
            h = stats[f][3]
            cv2.rectangle(backtorgb_mid, (x, y), (x + w, y + h), (0, 255, 0), 10)

        for f in range(1,numLabels):
            if f == max_h_index:
                continue
            
            x = stats[f][0]
            y = stats[f][1]
            w = stats[f][2]
            h = stats[f][3]
            cv2.rectangle(threshInv, (x, y), (x + w, y + h), (0, 0, 0), -1)

        x1 = stats[max_h_index][0]
        y1 = stats[max_h_index][1]
        w1 = stats[max_h_index][2]
        h1 = stats[max_h_index][3]
        bbox = [x1,y1,w1,h1]
        cx = int(centroids[max_h_index][0])
        cy = int(centroids[max_h_index][1])
        return bbox,x1,y1,w1,h1,cx,cy

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
        ###print(os.path.join(save_path, filename))
        ###print('Successfully generated and saved',filename)


    def improve_txbox(self,img,cx,cy):
        h, w = img.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        img_copy = img.copy()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (T, threshInv) = cv2.threshold(img_gray, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img_copy = cv2.blur(img_copy,(3,3))
        return
    
    def analize_text(self,img,x1,y1,w1,h1,save_path,f):
        TH = 0.05
        h, w = img.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        x,y = img.shape[:2]
        amp_x,amp_y,amp_x2,amp_y2 = round(x*(TH)),round(y*(TH)),round(x*(TH)),round(y*(TH))

        if (y1-amp_y)<0:
            y1 = 0
            amp_y = 0

        if (x1-amp_x)<0:
            x1 = 0
            amp_x = 0

        crop_text = img[y1-amp_y:y1+h1+amp_y2, x1-amp_x:x1+w1+amp_x2]
        img_gray = cv2.cvtColor(crop_text, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.blur(img_gray,(3,3))
        (T, threshInv) = cv2.threshold(img_gray, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        eroded = cv2.erode(threshInv,np.ones((2,2)))
        eroded = cv2.dilate(eroded,np.ones((2,2)))
        output = cv2.connectedComponentsWithStats(
            eroded, 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output

        #-----------
        number_white = np.sum(eroded == 255)
        number_black = np.sum(eroded == 0)
        #print('WHITE: ',number_white,'BLACK:',number_black)
        #-----------


        text = self.filter_text(eroded)      

        if number_white>number_black:
            eroded = np.invert(eroded)
            #print('Invertido')

        eroded = cv2.cvtColor(eroded, cv2.COLOR_GRAY2RGB)

        for f in range(1,numLabels):
            x = stats[f][0]
            y = stats[f][1]
            w = stats[f][2]
            h = stats[f][3]
            pixels = stats[f][4] # Area
            cv2.rectangle(eroded, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #print('TEXT:',text)
        
        return text

    def filter_text(self,eroded):
        text = pytesseract.image_to_string(eroded,config = '--psm 7 -c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        Count = 0
        result = []
        for k in text:
            if k.isupper():
                result.append(k)
        text_cleaned = re.sub(r"[^a-zA-Z0-9]+", ' ', text).replace("_","")
        text_cleaned = text_cleaned.split(' ')
        text_final = ''
        for txt in text_cleaned:
            txt = ''.join([i for i in txt if not i.isdigit()])
            if (len(txt)<3) or not text_cleaned[Count][0].isupper():
                txt = ''
            Count = Count + 1
            text_final = text_final +' '+ txt
            #print('TEXT:',txt)
        return text_final

    def text_distance(self,text_1, text_2, alg='levenshtein'):

        if text_1 is None or text_2 is None:
            return 1.0
        distance1 = TEXT_SIMILARITIES_ALG[alg](text_1, text_2)
        #print('SIM1:',distance1)
        distance = 1 / (1 + exp(-50*(distance1-0.05)))
        #print('SIM2:',distance)
        return 1 - distance,distance1

    def text_extraction(self,img,save_path,f):
        sum2 = self.pre_process(img)
        (T, threshInv) = cv2.threshold(sum2, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        bbox,x1,y1,w1,h1,cx,cy = self.search_elements(img,threshInv)
        backtorgb = self.gray2rgb(threshInv)
        text = self.analize_text(img,x1,y1,w1,h1,save_path,f)
        mask = self.generate_mask(img,bbox)
        backtorgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        self.improve_txbox(img,cx,cy)
        #----------------------------
        img_path = 'C:\\Users\\usuario\\Documents\\GitHub\\ABC\\CV_M1\\W3\\canvas_names.txt'
        text_txt = text
        txt_file = open(img_path,"a")
        txt_file.write(text_txt+'\n')
        #-------------------------
        if save_path is not None:
            self.save_mask(backtorgb,save_path,f)

        bbox[2],bbox[3] = bbox[0]+bbox[2],bbox[1]+bbox[3]
        ###print('BBOX: ',bbox)
        ###print('---------------------------------------------------------')
        return bbox,mask,text

    def compute_image_similarity(
        self, dataset, similarity_mode, query_img, text_extractor_method
    ):
        result = []
        _, _, text = self.text_extraction(query_img, None, None)
        for image in dataset.keys():
            _, distance = self.text_distance(dataset[image]["image_text"], text, similarity_mode)
            result.append([image, 1-distance])
        return result

                

valid_images = [".jpg"]
path = 'C:\\Users\\usuario\\Documents\\GitHub\\ABC\\CV_M1\\W3\\QSD1'
save_path = 'C:\\Users\\usuario\\Documents\\GitHub\\ABC\\CV_M1\\W3\\QSD1\\generated_text_masks'
if __name__ == "__main__":
    text_id = Text()
    for f in os.listdir(path):
        file_name = typex = os.path.splitext(f)[0]
        typex = os.path.splitext(f)[1]
        if typex.lower() not in valid_images:
            continue
        #path_in = path + '\\' + f
        path_in = os.path.join(path, f)
        #print(path_in)
        img = text_id.input_image(path_in)
        bbox,mask,text = text_id.text_extraction(img,save_path,file_name)
        print(f)
        print('TEXT:',text)
