from operator import le
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.histograms import histogram
from noise_remover import NoiseRemover

#Techniques like Wathershed, Canny, etc... Are considered "illegal"

MIN_AREA_IMG_RATIO = 1/15
MAX_N_FRAMES = 3
debug=False

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
        self.noise_remover = NoiseRemover()

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
        max_h_index2 = None
        output = cv.connectedComponentsWithStats(
            nogaps, 4, cv.CV_32S
        )
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
                if (w * h)/ (nogaps.shape[0] * nogaps.shape[1]) > MIN_AREA_IMG_RATIO:
                    print("primer cuadro")
                    print(x, w, h,y)
                    print(nogaps.shape[0], nogaps.shape[1])
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
                    if (w2 * h2)/ (nogaps.shape[0] * nogaps.shape[1]) > MIN_AREA_IMG_RATIO:
                        print("segundo cuadro")
                        print(x2, w2, h2,y2)
                        print(nogaps.shape[0], nogaps.shape[1])
                        #print('DADWSDAW',x2,y2)
                        #------------------------------ White
                        cv.rectangle(nogaps, (x2, y2), (x2 + w2, y2 + h2), (255, 255, 255), -1)
                        #------------------------------ White
                        #print(f,'SALTA1')
                continue
        
        if max_h_index2 is not None:
            if (w2 * h2)/ (nogaps.shape[0] * nogaps.shape[1]) > MIN_AREA_IMG_RATIO:
                if x2 < x:
                    tempx2 = x2
                    tempy2 = y2
                    tempw2 = w2
                    temph2 = h2
                    x2 = x
                    y2 = y
                    w2 = w
                    h2 = h 
                    x = tempx2
                    y = tempy2
                    w = tempw2
                    h = temph2
            else:
                x2 = -1
                y2 = -1
                w2 = -1
                h2 = -1
        cx = int(centroids[max_h_index][0])
        cy = int(centroids[max_h_index][1])
        #print('Centroids: X:',cx,'Y:',cy)
        #plt.imshow(cv.cvtColor(backtorgb_mid, cv.COLOR_BGR2RGB))
        #if x2 < x and max_h_index2 is not None:
        #    return nogaps,cx,cy,x2,y2,w2,h2,x,y,w,h
        return nogaps,cx,cy,x,y,w,h,x2,y2,w2,h2

    def gray2rgb(self,nogaps):
        #------------ Convert the image
        backtorgb = cv.cvtColor(nogaps,cv.COLOR_GRAY2RGB)
        #cv.rectangle(backtorgb, (x, y), (x + w, y + h), (0, 255, 0), 3)
        #plt.imshow(cv.cvtColor(backtorgb, cv.COLOR_BGR2RGB))
        return backtorgb

    def save_mask(self,backtorgb,mask,img,cx,cy,save_path,f):
        #------------ Save the mask and the image + mask
        #directory = save_path
        #os.chdir(directory)
        filename = os.path.splitext(f)[0]+'.png'
        file_path = os.path.join(save_path, filename)
        cv.imwrite(file_path, backtorgb)
        print('Successfully generated and saved',filename)

    """
        def crop_image(self,img,save_directory_croped,x,y,w,h,x2,y2,w2,h2,f):
            file_name = os.path.splitext(f)[0]
            #print('FILE NAME:',file_name)
            croped_img = img[y:y+h, x:x+w]
            #directory = save_directory_croped
            #os.chdir(directory)
            filename = 'crop_'+file_name+'.jpg'
            file_path = os.path.join(save_directory_croped, filename)
            cv.imwrite(file_path, croped_img)
            #cv.imwrite(filename, croped_img)
            print('Successfully generated and saved',filename)
            if x2 > 0:
                croped_img = img[y2:y2+h2, x2:x2+w2]
                #directory = save_directory_croped
                #os.chdir(directory)
                filename = 'crop_'+file_name+'_2'+'.jpg'
                file_path = os.path.join(save_directory_croped, filename)
                cv.imwrite(file_path, croped_img)
                #cv.imwrite(filename, croped_img)
                #print('Successfully generated and saved',filename)
    """

    def get_rectangle_to_crop(self, img, frames_pos):
        rows,cols = img.shape[0], img.shape[1]
        M = cv.getRotationMatrix2D((cols/2,rows/2),-frames_pos[4],1)
        img_rot = cv.warpAffine(img,M,(cols,rows))
        # rotate bounding box
        box = cv.boxPoints(((frames_pos[0],frames_pos[1]), (frames_pos[2],frames_pos[3]), -frames_pos[4]))
        pts = np.int0(cv.transform(np.array([box]), M))[0]    
        pts[pts < 0] = 0

        # crop
        img_crop = img_rot[pts[1][1]:pts[0][1], 
                            pts[1][0]:pts[2][0]]
        return img_crop

    def crop_image(self,img,save_directory_croped,frames_pos,f):
        file_name = os.path.splitext(f)[0]
        for idx, frame in enumerate(frames_pos):
            #x,y,w,h, angle = frame
            croped_img = self.get_rectangle_to_crop(img, frame)
            filename = 'crop_'+file_name+ '_' + str(idx+1) + '.jpg'
            file_path = os.path.join(save_directory_croped, filename)
            cv.imwrite(file_path, croped_img)
            #cv.imwrite(filename, croped_img)
            print('Successfully generated and saved',filename)
    
    def refine_angle(self, angle):
        if -88 < angle < -45:
            angle = -90 + abs(angle)
        elif -10 > angle >= -45:
            angle = abs(angle)
        elif angle < -88:
            angle = angle + 90
        return angle

    def compute_edges(self, img, sigma=0.9):
        median = np.median(img)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        edged = cv.Canny(img, 50, 170)
        # return the edged image
        return edged

    def sort_coord(self, box_cord):
        res = []
        new_box = []
        box_cord = box_cord.tolist()
        for box in box_cord:
            res.append(box[0] * box[1])
        new_box.append(box_cord[np.argmin(res)])
        new_box.append(box_cord[np.argmax(res)])
        box_cord.remove(new_box[0])
        box_cord.remove(new_box[1])
        if box_cord[0][0] > box_cord[1][0]:
            new_box.append(box_cord[0])
            new_box.append(box_cord[1])
        else:
            new_box.append(box_cord[1])
            new_box.append(box_cord[0])
        return [new_box[0], new_box[2], new_box[1], new_box[3]]
    
    def compute_angle(self, bottom_1, bottom_2):
        import math
        tangent=np.sqrt((bottom_1[0]-bottom_2[0])**2 + (bottom_1[1]-bottom_2[1])**2)
        angle = np.arccos(
            (bottom_2[0]-bottom_1[0]) / tangent
        )
        return math.degrees(angle)

    def compute_background(self, img):
        resulting_frame_pos = []
        list_boxes_sorted = []
        # get a blank canvas for drawing contour on and convert img to grayscale
        canvas = np.zeros(img.shape, np.uint8)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #smoth the image
        blur = cv.GaussianBlur(gray,(7, 7), 0)
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        gray = cv.filter2D(blur, -1, sharpen_kernel)
        # apply Canny edge detection using a wide threshold, tight
        # threshold, and automatically determined threshold
        edges = self.compute_edges(gray)

        kernel = cv.getStructuringElement(cv.MORPH_RECT,(13,13))
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=3)

        #kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(15,15))
        #edges = cv.dilate(edges, kernel)
         
        #edges = cv.erode(edges, None)
        _ ,contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        # find the main island (biggest area)
        contours = sorted(contours, key=cv.contourArea, reverse=True)
        contours = contours[:MAX_N_FRAMES]
        contours = sorted(contours, key=lambda contours: cv.boundingRect(contours)[0])
        for cnt in contours:
            img_to_draw = img.copy()
            max_area = cv.contourArea(cnt)
            # define main island contour approx. and hull
            #print(approx)
            # show the images
            if max_area/ (img.shape[0] * img.shape[1]) > MIN_AREA_IMG_RATIO:
                #epsilon = 0.1*cv.arcLength(cnt,True)
                #approx = cv.approxPolyDP(cnt,epsilon,True)
                hull = cv.convexHull(cnt)
                #cv.imshow("Original", img)
                #cv.waitKey(0)
                #cv.drawContours(img_to_draw, cnt, -1, (0, 255, 0), 3)
                pos,lenght,angle = cv.minAreaRect(hull)
                box = cv.boxPoints((pos,lenght,angle))
                box = np.int0(box)
                x,y= pos
                w,h=lenght
                box_sorted = self.sort_coord(box)
                angle = self.refine_angle(angle)
                mod_angle = self.compute_angle(box_sorted[2], box_sorted[3])
                #cv.drawContours(img_to_draw, cnt, -1, (0, 255, 0), 3)
                #x,y,w,h = cv.boundingRect(hull)
                resulting_frame_pos.append([x,y,w,h,angle])
                box_final = [mod_angle,box_sorted]
                list_boxes_sorted.append(box_final)
                #cv.drawContours(img_to_draw, cnt, -1, (0, 0, 255), 3)
                cv.drawContours(img_to_draw,[box],0,(0,255,0),2)
                cv.drawContours(canvas,[box],0,(255, 255, 255), -1)
                #cv.rectangle(img_to_draw,(x,y),(x+w,y+h),(0,255,0),2)
                #cv.rectangle(canvas,(x,y),(x+w,y+h),(255, 255, 255), -1)
                #cv.drawContours(img_to_draw, rct, -1, (0, 0, 255), 3)
                #cv.drawContours(canvas, approx, -1, (0, 0, 255), 3)
                # cv.drawContours(canvas, hull, -1, (0, 0, 255), 3) # only displays a few points as well.
                if debug:
                    title ="Display frame"
                    cv.namedWindow( title, cv.WINDOW_AUTOSIZE)
                    #cv.resizeWindow("Display frame", 500, 500)
                    cv.namedWindow(title)
                    cv.moveWindow(title, 500, 500)
                    cv.imshow(title, cv.resize(img_to_draw, (200, 200)) )
                    cv.waitKey(0)
                    cv.destroyAllWindows()
        if debug:
            title ="Display2 frame"
            cv.namedWindow( title, cv.WINDOW_AUTOSIZE)
            #cv.resizeWindow("Display frame", 500, 500)
            cv.namedWindow(title)
            cv.moveWindow(title, 500, 500)
            cv.imshow(title, cv.resize(canvas, (200, 200)) )
            cv.waitKey(0)
            cv.destroyAllWindows()
        # If no detected frame we will take all the image
        if len(resulting_frame_pos) < 1:
            resulting_frame_pos.append([0, 0, img.shape[1], img.shape[0], 0])
        return resulting_frame_pos, canvas, list_boxes_sorted

    def background_remover(self,path,save_path,save_path_croped,f):
        img = self.input_image(path)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
        if  self.noise_remover.is_noisy_img(img_gray):
            denoised_img = self.noise_remover.remove_noise(img, "median", 5)
            img = denoised_img
        # smoth the image
        frames_pos, mask, list_boxes_sorted = self.compute_background(img)
        #simplifyed_img, histogram, result = self.simplify_irrelevant(img)
        #objects_img, mask = self.make_bin_and_objects(simplifyed_img)
        #connected_img,cx,cy,x,y,w,h,x2,y2,w2,h2 = self.connected_componets2(objects_img)
        #final_mask = self.gray2rgb(mask)
        self.save_mask(mask,mask,img,"","",save_path,f)
        self.crop_image(img,save_path_croped,frames_pos,f)
        #return final_mask,x,y,w,h,x2,y2,w2,h2
        return mask, list_boxes_sorted 

valid_images = [".jpg"]
load_directory = 'C:\\Users\\JQ\\Documents\\GitHub\\ABC\\CV_M1\\W2\\QSD2\\'
save_direcory = 'C:\\Users\\JQ\\Documents\\GitHub\\ABC\\CV_M1\\W2\\QSD2\\generated_masks'
save_directory_croped = 'C:\\Users\\JQ\\Documents\\GitHub\\ABC\\CV_M1\\W2\\QSD2\\croped'
load_directory = "datasets/qsd1_w5/"
save_direcory = 'datasets/qsd1_w5/generated_masks'
save_directory_croped = 'datasets/qsd1_w5/cropped'
if __name__ == "__main__":
    import results
    result = results.ground_truth("datasets/qsd1_w5/frames.pkl")
    idx = 0
    museum = Canvas()
    for f in sorted(os.listdir(load_directory)):        
        ##print(f)
        #f = "00010.jpg"
        file_name = typex = os.path.splitext(f)[0]
        typex = os.path.splitext(f)[1]
        ##print(typex)
        if typex.lower() not in valid_images:
            continue
        print(f)
#        _,x,y,w,h,x2,y2,w2,h2 = museum.background_remover(load_directory + f,save_direcory,save_directory_croped ,file_name)

        mask, list_boxes_sorted  = museum.background_remover(load_directory + f,save_direcory,save_directory_croped ,file_name)
        print("Result")
        print(result[idx])
        print("Guess")
        print(list_boxes_sorted)
        idx = idx + 1

        #print(frames_pos)
