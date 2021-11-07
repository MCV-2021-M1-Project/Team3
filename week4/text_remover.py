import numpy as np
import cv2 
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import pytesseract
import string
import re
import textdistance
import random
import pandas as pd
import sys
from tqdm import tqdm
from math import exp
from noise_remover import NoiseRemover

pd.options.display.max_rows = 4000
#pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\usuario\AppData\\Local\\Tesseract-OCR\\tesseract'
if sys.platform == 'win32':
    pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract'
noise_remover = NoiseRemover()

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

    def remove_noise(self,img,type='median'):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        if  noise_remover.is_noisy_img(img_gray):
            denoised_img = noise_remover.remove_noise(img, type, 3)
            return denoised_img
        else:
            return img

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

    def search_elements2(self,img,thresh):
        save_path = 'C:\\Users\\user\\Documents\\GitHub\\ABC\\CV_M1\\W4\\QSD1\\generated_text_masks\\AAA.jpg'
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            epsilon = 0.01*cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(img, [approx], 0, (0), 3)
            # Position for writing text
            x,y = approx[0][0]

            if len(approx) == 3:
                cv2.putText(img, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0,2)
            elif len(approx) == 4:
                cv2.putText(img, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0,2)
            elif len(approx) == 5:
                cv2.putText(img, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0,2)
            elif 6 < len(approx) < 15:
                cv2.putText(img, "Ellipse", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0,2)
            else:
                cv2.putText(img, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0,2)
        cv2.imwrite(save_path,img)
        return 

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
    
    def find_if_close(self,cnt1,cnt2):
        row1,row2 = cnt1.shape[0],cnt2.shape[0]
        for i in range(row1):
            for j in range(row2):
                dist = np.linalg.norm(cnt1[i]-cnt2[j])
                if abs(dist) < 150 :
                    return True
                elif i==row1-1 and j==row2-1:
                    return False


    def augm_coords(self,img,x1,y1,x2,y2,percentage1,percentage2):
        
        h, w = img.shape[:2]
        x,y = img.shape[:2]

        amp_x,amp_y,amp_x2,amp_y2 = round(x*(percentage2)),round(y*(percentage1)),round(x*(percentage2)),round(y*(percentage1))

        if (y1-amp_y)<0:
            y1 = 0
            amp_y = 0

        if (x1-amp_x)<0:
            x1 = 0
            amp_x = 0

        crop_text = img[y1-amp_y:y2+amp_y2, x1-amp_x:x2+amp_x2]
        return crop_text

    def analize_text(self,img,x1,y1,w1,h1,save_path,f):
        TH3 = 0.05
        h, w = img.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        x,y = img.shape[:2]
        amp_x,amp_y,amp_x2,amp_y2 = round(x*(TH3)),round(y*(TH3)),round(x*(TH3)),round(y*(TH3))

        if (y1-amp_y)<0:
            y1 = 0
            amp_y = 0

        if (x1-amp_x)<0:
            x1 = 0
            amp_x = 0

        crop_text = img[y1-amp_y:y1+h1+amp_y2, x1-amp_x:x1+w1+amp_x2]

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS,(6,6))
        kernelPen = cv2.getStructuringElement(cv2.MORPH_RECT, (3,5))
        
        image = crop_text
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        kernelTH = np.ones((30, 30), np.uint8) 
        img_TH = cv2.morphologyEx(crop_text, cv2.MORPH_TOPHAT, kernelTH)
        img_BH = cv2.morphologyEx(crop_text, cv2.MORPH_BLACKHAT, kernelTH)

        TH = 150
        img_TH[(img_TH[:,:,0] < TH) | (img_TH[:,:,1] < TH) | (img_TH[:,:,2] < TH)] = (0,0,0)

        img_BH[(img_BH[:,:,0] < TH) | (img_BH[:,:,1] < TH) | (img_BH[:,:,2] < TH)] = (0,0,0)
        img_sum = img_TH + img_BH

        kernelzp = np.ones((1, int(img.shape[1] / 30)), np.uint8) 
        img_TH = cv2.dilate(img_TH, kernelzp, iterations=1) 
        img_TH = cv2.erode(img_TH, kernelzp, iterations=1) 
        img_BH = cv2.dilate(img_TH, kernelzp, iterations=1) 
        img_BH = cv2.erode(img_TH, kernelzp, iterations=1) 
        img_sum = img_TH + img_BH


        img_sum = cv2.cvtColor(img_sum, cv2.COLOR_BGR2GRAY)
        ret,img_sum = cv2.threshold(img_sum,127,255,0)
        plt.imshow(cv2.cvtColor(img_sum, cv2.COLOR_BGR2RGB))
        img_sum = cv2.morphologyEx(img_sum, cv2.MORPH_OPEN, kernelPen)
        cnts,hier = cv2.findContours(img_sum, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text1 = ''
        text2 = ''
        bbox = [x1,y1,w1,h1]
        if len(cnts)>1:
            cnts = np.concatenate(cnts)
            # Determine and draw bounding rectangle
            x, y, w, h = cv2.boundingRect(cnts)
            bbox = [x1,y1,w1,h1]
            cv2.rectangle(img_sum, (x, y), (x + w - 1, y + h - 1), 255, 2)
            img_sum = self.augm_coords(crop_text,x,y,x+w,y + h,0.03,0.06)
            img_sum = cv2.cvtColor(img_sum, cv2.COLOR_BGR2GRAY)
            (T, img_sum) = cv2.threshold(img_sum, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            text1 = pytesseract.image_to_string(img_sum,config = '--psm 7 -c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
            text2 = self.filter_text(img_sum)
            if 1 == 0: 
                number_white = np.sum(img_sum == 255)
                number_black = np.sum(img_sum == 0)

                text = self.filter_text(img_sum)      

                if number_white>number_black:
                    eroded = np.invert(img_sum)
                    #print('Invertido')

        
        img_gray = cv2.cvtColor(crop_text, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.blur(img_gray,(3,3))
        (T, threshInv) = cv2.threshold(img_gray, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        eroded = cv2.erode(threshInv,np.ones((2,2)))
        eroded = cv2.dilate(eroded,np.ones((2,2)))
        output = cv2.connectedComponentsWithStats(
            eroded, 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        number_white = np.sum(eroded == 255)
        number_black = np.sum(eroded == 0)
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
        

        return text,text1,text2,bbox

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
        distance = 1 / (1 + exp(-50*(distance1-0.05)))
        return 1 - distance,distance1

    def text_extraction(self,img,save_path,f):
        img = self.remove_noise(img)
        plt.imshow(img)

        sum2 = self.pre_process(img)
        (T, threshInv) = cv2.threshold(sum2, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        w=img.shape[1]
        h=img.shape[0]
        a = round(h/100)
        b = round(w/70)
        a1 = round(h/150)
        b1 = round(w/120)
        kernel = np.ones((a,b),np.uint8)
        kernel1 = np.ones((a1,b1),np.uint8)

        opening = cv2.morphologyEx(threshInv, cv2.MORPH_OPEN, kernel)
        
        bbox,x1,y1,w1,h1,cx,cy = self.search_elements(img,opening)
        backtorgb = self.gray2rgb(threshInv)
        text,text1,text2,bbox = self.analize_text(img,x1,y1,w1,h1,save_path,f)
        mask = self.generate_mask(img,bbox)
        backtorgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        self.improve_txbox(img,cx,cy)

        if save_path is not None:
            self.save_mask(backtorgb,save_path,f)

        bbox[2],bbox[3] = bbox[0]+bbox[2],bbox[1]+bbox[3]

        ###print('BBOX: ',bbox)
        ###print('---------------------------------------------------------')
        return bbox,mask,text,text1,text2

    def compute_image_similarity(
        self, dataset, similarity_mode, query_img, text_extractor_method
    ):
        result = []
        _, _, text,text1,text2 = self.text_extraction(query_img, None, None)
        for image in dataset.keys():
            _, distance = self.text_distance(dataset[image]["image_text"], text, similarity_mode)
            result.append([image, 1-distance])
        return result

    def compute_image_similarity(self,dataset, similarity_mode, query_img,BBDD_GT_Dataset,file_name,txt_save_path, text_extractor_method):
        text,text1,text2 = self.text_extraction(query_img, None, None)
        textS=[text,text1,text2]
        dist_obt1 = 0
        text_obt1 = ''

        for tex in textS:
            for ground_T in BBDD_GT_Dataset:
                dist,dist1 = text_id.text_distance(ground_T,tex)

                if dist1>dist_obt1:
                    dist_obt1 = dist1
                    text_obt1 = ground_T
                else:
                    dist_obt1 = dist_obt1
        result = []
        for image in dataset.keys():
            _, distance = self.text_distance(dataset[image]["image_text"], dist_obt1, similarity_mode)
            result.append([image, 1-distance])
        print('-----------------------------------------------------------------------------------------')
        print('DISTANCE:', dist_obt1,'---->',text_obt1,'Result:',result)
        self.generate_txt(text_obt1,file_name,txt_save_path)
        return result

    def generate_txt(self,text_obt1,file_name,txt_save_path): #file_name: 00000.jpg or 00000_2.jpg
        txt_save_path = os.path.join(txt_save_path,file_name)
        file_name = file_name[:5]
        txt_save_path = 'C:\\Users\\user\\Documents\\GitHub\\ABC\\CV_M1\\W4\\QSD1\\txt\\'+file_name+'.txt'
        txt_file = open(txt_save_path,'a')
        txt_file.write(text_obt1+'\n')
        print('-----------------------------------------------------------------------------------------')
        return


valid_images = [".jpg"]
valid_text = ['.txt']
path = 'C:\\Users\\user\\Documents\\GitHub\\ABC\\CV_M1\\W4\\QSD1'
path = 'C:\\Users\\user\\Documents\\GitHub\\ABC\\CV_M1\\W4\\QSD1'
save_path = 'C:\\Users\\user\\Documents\\GitHub\\ABC\\CV_M1\\W4\\QSD1\\generated_text_masks2'
path_dataset = 'C:\\Users\\user\\Documents\\GitHub\\ABC\\CV_M1\\data\\BBDD'


if __name__ == "__main__":
    x = 0
    sim_sum = 0
    text_id = Text()
    #------------------------- Cargar Dataset BBDD_txt------------
    BBDD_GT_Dataset = []
    os.chdir(path_dataset)
    for BBDD_GT in tqdm(os.listdir(path_dataset)):
        GT_txt_path = os.path.join(BBDD_GT)
        file_name_txt = typex_txt = os.path.splitext(BBDD_GT)[0]
        typex_txt = os.path.splitext(BBDD_GT)[1]
        if typex_txt.lower() not in valid_text:
            continue
        with open(GT_txt_path, "r") as f:
            BBDD_GT_Names = f.read().split(',')
            for Names_GT in BBDD_GT_Names:
                Names_GT_Net = Names_GT.translate({ ord(c): None for c in "')(\n" })
                BBDD_GT_Dataset.append(Names_GT_Net)
    #------------------------------------------------------------
    #print(BBDD_GT_Dataset)
    os.chdir(path)

    for f in tqdm(os.listdir(path)):
        
        if x >= 50:
            #print('Skipped')
            continue
        #print('F:',f)
        #print('PATH:',path)
        file_name = typex = os.path.splitext(f)[0]
        typex = os.path.splitext(f)[1]
        ##print(typex)
        if typex.lower() not in valid_images:
            continue
        path_in = os.path.join(path, f)
        print('F:',f)
        #print(path_in)
        img = text_id.input_image(path_in)
        bbox,mask,text,text1,text2 = text_id.text_extraction(img,save_path,file_name)
        #-------------------------------------------
        textS=[text,text1,text2]
        GT_path = path_in.replace("jpg", "txt")
        dist_obt1 = 0
        text_obt1 = ''

        for tex in textS:
            for ground_T in BBDD_GT_Dataset:
                dist,dist1 = text_id.text_distance(ground_T,tex)

                if (dist1>dist_obt1) and (ground_T != '') :
                    dist_obt1 = dist1
                    text_obt1 = ground_T
                    print('ground_T',ground_T,'dist:',dist1)
                else:
                    dist_obt1 = dist_obt1
                
        print('DISTANCE:', dist_obt1,'---->',text_obt1)
        #------------------------------------------
        #----------------------------
        file_name = file_name[:5]
        img_path = 'C:\\Users\\user\\Documents\\GitHub\\ABC\\CV_M1\\W4\\QSD1\\txt\\'+file_name+'.txt'
        txt_file = open(img_path,'a')
        txt_file.write(text_obt1+'\n')
        print('-----------------------------------------------------------------------------------------')
        #-------------------------
        x += 1

    print(x)
    print('FINAL: ',sim_sum/x)

