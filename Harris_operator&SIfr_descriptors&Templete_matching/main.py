from PyQt5 import QtWidgets, QtCore, QtGui
import qimage2ndarray as qtimg
from scipy import ndimage
import numpy
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import qimage2ndarray
from ui import Ui_MainWindow
from scipy import signal
from scipy.signal import convolve2d
import ntpath
import copy
import sys
import cv2
from PIL import Image, ImageDraw
from math import sqrt, pi, cos, sin, atan2,exp
import numpy as np

class App(QtWidgets.QMainWindow):
    
    path1 = ""
    path2 = ""
    path3 = ""
    
    
    def __init__(self):
        super(App, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.ui.b1.clicked.connect(self.loaddata1)
        self.ui.b3.clicked.connect(self.loaddata2)
        self.ui.b4.clicked.connect(self.loaddata3)
        self.ui.b2.clicked.connect(self.apply_harris)
        self.ui.b5.clicked.connect(self.matching)
    
    
    
    def loaddata1(self):
        filename = QFileDialog.getOpenFileName(self)
        if filename[0]:
            self.path1 = filename[0]
        
        image = cv2.imread(self.path1)
        self.ui.w1.show()
        self.ui.w1.setImage(np.rot90(image,1))
        
    def loaddata2(self):
        filename = QFileDialog.getOpenFileName(self)
        if filename[0]:
            self.path2 = filename[0]
        
        image = cv2.imread(self.path2)
        self.ui.w3.show()
        self.ui.w3.setImage(np.rot90(image,1))
        self.ui.w5.show()
        self.ui.w5.setImage(np.rot90(image,1))
        
    def loaddata3(self):
        filename = QFileDialog.getOpenFileName(self)
        if filename[0]:
            self.path3 = filename[0]
        
        image = cv2.imread(self.path3)
        self.ui.w4.show()
        self.ui.w4.setImage(np.rot90(image,1))
        self.ui.w6.show()
        self.ui.w6.setImage(np.rot90(image,1))
    
    def apply_harris(self):
        image = cv2.imread(self.path1)
        img_plot = np.copy(image)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray_img = np.float32(gray)
        points = self.harris(gray_img)
        for point in points:
            cv2.circle(img_plot, (point[1],point[0]), 5 , [255,0,255], -1)
        cv2.imwrite("out/harris_points.png", img_plot)
        out_harris = cv2.imread("out/harris_points.png")
        self.ui.w2.show()
        self.ui.w2.setImage(np.rot90(out_harris,1))
    
    def harris(self,gray_img):
        
        smothed_image = signal.convolve2d(gray_img, self.gaussian_kernel2d(7,1) ,'same')
        Ix =  cv2.Sobel(smothed_image,cv2.CV_64F,1,0,ksize=3)
        Iy = cv2.Sobel(smothed_image,cv2.CV_64F,0,1,ksize=3)
        Ixx =  np.multiply( Ix, Ix) 
        Iyy =  np.multiply( Iy, Iy)
        Ixy =  np.multiply( Ix, Iy)
        Ixx_hat = signal.convolve2d( Ixx , self.box_filter(3) ,'same') 
        Iyy_hat = signal.convolve2d( Iyy , self.box_filter(3) ,'same') 
        Ixy_hat = signal.convolve2d( Ixy , self.box_filter(3) ,'same')
        k = 0.04
        detM = np.multiply(Ixx_hat,Iyy_hat) - np.multiply(Ixy_hat,Ixy_hat) 
        trM = Ixx_hat + Iyy_hat
        R = detM - (k * (trM**2))
        corners = R > 0.01*R.max()
        points_x = np.where(corners==True)[0].reshape(len(np.where(corners==True)[0]),-1)
        points_y = np.where(corners==True)[1].reshape(len(np.where(corners==True)[1]),-1)
        points = np.concatenate((points_x,points_y),axis = 1)
        return points
        
    def apply_sift(self):  
        
        image1 = cv2.imread(self.path2)
        image1 = cv2.resize(image1, (1000,1000), interpolation = cv2.INTER_AREA)
        gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
        gray_img1 = np.float32(gray1)
        points1 = self.harris(gray_img1)
        discreptors1_points ,discreptors1  = self.sift(points1 , gray_img1,[8,980])
        
        image2 = cv2.imread(self.path3)
        image2 = cv2.resize(image2, (300,300), interpolation = cv2.INTER_AREA)
        gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
        gray_img2 = np.float32(gray2)
        points2 = self.harris(gray_img2)
        discreptors2_points ,discreptors2 = self.sift(points2 , gray_img2,[8,280])
        
        return image1 , image2 , discreptors1_points , discreptors1 , discreptors2_points , discreptors2
        
    def sift(self, points,gray_img,limit):
        direction_list = [0, 45, 90, 135,180,225,270,315,360]
        gaussian_img  = convolve2d( gray_img , self.gaussian_kernel2d(7,1.5) , 'same', 'symm') 
        gauss_kernal = self.gaussian_kernel2d(std = 1.5, kernlen = 16)   
        gx,gy,magnitude,direction = self.sift_gradient(gaussian_img)

        ## orinatation & rotate 
        magnitude = self.patch_orientation(direction,magnitude,limit,points)

        ## Gaussian with patch sigma=1.5
        for x , y in points:
            if((x > limit[0] and x < limit[1] ) and ( y > limit[0] and y < limit[1])):
                lT_point_x = x - 7 
                lT_point_y = y - 7
                magnitude[lT_point_x:lT_point_x+16,lT_point_y:lT_point_y+16] = magnitude[lT_point_x:lT_point_x+16,lT_point_y:lT_point_y+16] * gauss_kernal

        ## Get the 4*4 parts & compute descreptors
        discreptors_points = []
        flag2 = 0
        for x , y in points:
            if((x > 15 and x < 980 ) and ( y > 15 and y < 980)):
                flag1 = 0

                for part in self.get_4parts(magnitude , x , y):
                    if (flag1 == 0):
                        hist_arr = np.histogram(part, bins = direction_list)[0]
                        flag1 = 1
                    else:
                        hist_arr = np.concatenate((hist_arr,np.array(np.histogram(part, bins = direction_list)[0])))
                if(flag2 == 0):
                    discreptors = hist_arr.reshape(128,-1)
                    flag2 = 1
                else:
                    discreptors = np.concatenate((discreptors ,hist_arr.reshape(128,-1)),axis = 1)
                discreptors_points.append((x,y)) 

        return discreptors_points,discreptors.T
        
    
    def matching(self):
        
        image1 , image2 , discreptors1_points , discreptors1 , discreptors2_points , discreptors2 = self.apply_sift()
        match_arr = np.zeros((discreptors1.shape[0],discreptors2.shape[0]))
        for i in range(discreptors1.shape[0]):
            for j in range(discreptors2.shape[0]):
                match_arr[i,j] = np.sum(np.square(discreptors1[i] - discreptors2[j]))
        ## SSD match      
        plot_img1 = np.copy(image1)
        plot_img2 = np.copy(image2)

        point_des_1 = np.where(match_arr < np.min(match_arr)+80)[0]
        point_des_2 = np.where(match_arr < np.min(match_arr)+80)[1]

        for index in point_des_1:
            cv2.circle(plot_img1, (discreptors1_points[index][1],discreptors1_points[index][0]), 2, [255,0,255], -1)
        cv2.imwrite("out/SSD_match_main_image.png", plot_img1)

        for index in point_des_2:
            cv2.circle(plot_img2, (discreptors2_points[index][1],discreptors2_points[index][0]), 1, [255,0,255], -1)
        cv2.imwrite("out/SSD_match_temp_image.png", plot_img2)
        
        out_main_ssd = cv2.imread("out/SSD_match_main_image.png")
        self.ui.w3.show()
        self.ui.w3.setImage(np.rot90(out_main_ssd,1))
        out_temp_ssd = cv2.imread("out/SSD_match_temp_image.png")
        self.ui.w4.show()
        self.ui.w4.setImage(np.rot90(out_temp_ssd,1))
        
        # Normalize cross correlation match 
        match_arr_norm = np.zeros((discreptors1.shape[0],discreptors2.shape[0]))
        for i in range(discreptors1.shape[0]):
            for j in range(discreptors2.shape[0]):
                num =((np.sum(np.square(self.norm_data(discreptors1[i]))))*(np.sum(np.square(self.norm_data(discreptors2[j])))))
                dom = (np.sum(self.norm_data(discreptors1[i]) * self.norm_data(discreptors2[j])))
                match_arr_norm[i,j] = num / dom if dom else 0
        plot_n1 = np.copy(image1)
        plot_n2 = np.copy(image2)
        
        point_norm_1 = np.where(match_arr_norm > np.max(match_arr_norm)-(4*pow(10,19)))[0]
        point_norm_2 = np.where(match_arr_norm > np.max(match_arr_norm)-(4*pow(10,19)))[1]
        
        print(point_norm_1.shape)
        
        for index in point_norm_1:
            cv2.circle(plot_n1, (discreptors1_points[index][1],discreptors1_points[index][0]), 2, [255,0,255], -1)
        cv2.imwrite("out/norm_match_main_image.png", plot_n1)

        for index in point_norm_2:
            cv2.circle(plot_n2, (discreptors2_points[index][1],discreptors2_points[index][0]), 1, [255,0,255], -1)
        cv2.imwrite("out/norm_match_temp_image.png", plot_n2)
        
        out_main_norm = cv2.imread("out/norm_match_main_image.png")
        self.ui.w5.show()
        self.ui.w5.setImage(np.rot90(out_main_norm,1))
        out_temp_norm = cv2.imread("out/norm_match_temp_image.png")
        self.ui.w6.show()
        self.ui.w6.setImage(np.rot90(out_temp_norm,1))
        
        
    def gaussian_kernel1d(self,kernlen=7, std=1.5):
        kernel1d = signal.gaussian(kernlen, std=std)
        kernel1d = kernel1d.reshape(kernlen, 1)
        return kernel1d / kernel1d.sum()

    def gaussian_kernel2d(self,kernlen=7, std=1.5):
        gkern1d = self.gaussian_kernel1d(kernlen,std)
        gkern2d = np.outer(gkern1d, gkern1d)
        return gkern2d

    def box_filter(self,f):
        kernel = np.ones((f,f)) / (f**2)
        return kernel
    
    def sift_gradient(self,img):
        dx = np.array([[-1,0,1],
                      [-2,0,2],
                      [-1,0,1]])
        dy = dx.T
        gx = signal.convolve2d( img , dx , boundary='symm', mode='same' )
        gy = signal.convolve2d( img , dy , boundary='symm', mode='same' )
        magnitude = np.sqrt( gx * gx + gy * gy )
        direction = np.rad2deg( np.arctan2( gy , gx )) % 360
        return gx,gy,magnitude,direction
    
    def get_4parts(self,img,x,y):
        parts = []
        counter = 0
        lT_point_x = x - 7 
        lT_point_y = y - 7
        for i in range(16):
            parts.append(img[lT_point_x+counter:lT_point_x+counter+4,lT_point_y+counter:lT_point_y+counter+4])
            counter = counter + 4 
        return parts
    
    def patch_orientation(self,direction,magnitude,limit,points):
        direction_list = [0, 45, 90, 135,180,225,270,315,360]
        points_orientation = []
        for x , y in points:
            if((x > limit[0] and x < limit[1] ) and ( y > limit[0] and y < limit[1])):
                lT_point_x = x - 7 
                lT_point_y = y - 7
                window = direction[lT_point_x:lT_point_x+16,lT_point_y:lT_point_y+16]
                orientation = direction_list[np.argmax(np.histogram(window, bins = direction_list)[0])]
                magnitude[lT_point_x:lT_point_x+16,lT_point_y:lT_point_y+16] = ndimage.rotate(magnitude   [lT_point_x:lT_point_x+16,lT_point_y:lT_point_y+16], orientation, reshape=False)
                return magnitude
            
    def norm_data(self,data):

        mean_data=np.mean(data)
        std_data=np.std(data, ddof=1)
        return (data-mean_data)/(std_data)
    
def main():
    app = QtWidgets.QApplication(sys.argv)
    application = App()
    application.show()
    app.exec_()
if __name__ == "__main__":
    main()
