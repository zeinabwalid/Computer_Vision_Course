from PyQt5 import QtWidgets, QtCore, QtGui
import qimage2ndarray as qtimg
import numpy
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import qimage2ndarray
from gui import Ui_MainWindow
from scipy.signal import convolve2d
import ntpath
import copy
import sys
import cv2
from PIL import Image, ImageDraw
from math import sqrt, pi, cos, sin, atan2,exp
from collections import defaultdict
import numpy as np

class App(QtWidgets.QMainWindow):
    filter_tab_input_img = None
    hough_img = None
    path_hough = ""
    path_snake = ""
    path_canny = ""

    def __init__(self):
        super(App, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.label_active_contours_input.setMouseTracking(True)
        self.ui.label_hough_input.setScaledContents(True)
        self.ui.label_hough_output.setScaledContents(True)
        self.ui.label_active_contours_input.mousePressEvent = self.mouse_coordinates


        self.labelWidth = self.ui.label_active_contours_input.frameGeometry().width()
        self.labelHeight = self.ui.label_active_contours_input.frameGeometry().height()
        self.my_points= []
        ## Texts input
        self.alpha = self.ui.textEdit_alpha
        self.beta = self.ui.textEdit_beta
        self.gamma = self.ui.textEdit_gamma
        self.iterations = self.ui.textEdit_iterations
        self.line_thr = self.ui.textEdit_7
        self.circule_thr = self.ui.textEdit_8

        # Variables
        self.img_countour = None
        self.img_hough = None
        self.img_canny = None
        self.imagenew = None


        # Actions

        self.ui.pushButton_active_contours_load.clicked.connect(self.load_active_contour)
        self.ui.pushButton_canny_load.clicked.connect(self.load_canny)
        self.ui.pushButton_apply_canny.clicked.connect(self.canny)
        self.ui.pushButton_clear_anchors.clicked.connect(self.set_points)
        self.ui.pushButton_apply_active_contours.clicked.connect(self.apply_active_contour)
        self.ui.pushButton_reset_anchors.clicked.connect(self.reset_points)

        self.ui.pushButton_apply_hough.clicked.connect(self.apply_hough)

        self.ui.pushButton_hough_load.clicked.connect(self.load_hough)

    
    def apply_hough(self):
        if(self.ui.checkBox_lines.isChecked()):
            self.hough_line()
        elif(self.ui.checkBox_circles.isChecked()):
            self.hough_circule()

    def hough_line(self):
        hough_img = cv2.resize(self.img_hough,(100,100), interpolation = cv2.INTER_AREA)
        shapes_grayscale = cv2.cvtColor(hough_img, cv2.COLOR_RGB2GRAY)
        shapes_blurred = cv2.GaussianBlur(shapes_grayscale, (5, 5), 1.5)
        canny_edges = cv2.Canny(shapes_blurred, 100, 200)
        H, rhos, thetas = self.hough_lines_acc(canny_edges)
        indicies, H = self.hough_peaks(H, int(self.line_thr.toPlainText()), nhood_size=11)
        self.hough_lines_draw(hough_img, indicies, rhos, thetas)
        cv2.imwrite("imgs/hough_line.jpg", hough_img)
        hough_out = cv2.imread('imgs/hough_line.jpg')
        qimg = qtimg.array2qimage(hough_out)
        self.ui.label_hough_output.setPixmap(QPixmap(qimg))
    
    def hough_circule(self):
        input_image = Image.open(self.path_hough)
        output_image = Image.new("RGB", input_image.size)
        output_image.paste(input_image)
        draw_result = ImageDraw.Draw(output_image)
        rmin = int(self.circule_thr.toPlainText())
        rmax = 20
        steps = 100
        threshold = 0.4
        points = []
        for r in range(rmin, rmax + 1):
            for t in range(steps):
                points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))
        acc = defaultdict(int)
        for x, y in self.canny_edge_detector(input_image):
            for r, dx, dy in points:
                a = x - dx
                b = y - dy
                acc[(a, b, r)] += 1
        circles = []
        for k, v in sorted(acc.items(), key=lambda i: -i[1]):
            x, y, r = k
            if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in                circles):
                circles.append((x, y, r))
        for x, y, r in circles:
            draw_result.ellipse((x-r, y-r, x+r, y+r), outline=(255,255,0,0))
        output_image.save("imgs/hough_circule.jpg")
        hough_out = cv2.imread('imgs/hough_circule.jpg')
        qimg = qtimg.array2qimage(hough_out)
        self.ui.label_hough_output.setPixmap(QPixmap(qimg)) 
    
    
    def reset_points(self):
        self.my_points = []
        image = cv2.imread(self.path_snake) 
        image = cv2.resize(image,(200,200), interpolation = cv2.INTER_AREA)
        qimg = qtimg.array2qimage(image)
        self.ui.label_active_contours_output.setPixmap(QPixmap(qimg))
        self.ui.label_active_contours_input.setPixmap(QPixmap(qimg))
    
    def set_points(self):
        image = cv2.imread(self.path_snake) 
        image = cv2.resize(image,(200,200), interpolation = cv2.INTER_AREA)
        img = np.copy(image)
        red = [0,0,255]
        points = np.array(self.my_points)
        x = points[:,0].reshape(len(self.my_points),1).astype(int)
        y = points[:,1].reshape(len(self.my_points),1).astype(int)
        points = np.concatenate((x,y),axis = 1)
        copy_points = np.copy(points)
        image_out = np.copy(image)
        for point in copy_points:
            cv2.circle(image_out, (point[0],point[1]), 2, [200,0,0], -1)
        cv2.imwrite("imgs/snake_points.jpg", image_out)
        snake_out = cv2.imread('imgs/snake_points.jpg')
        qimg = qtimg.array2qimage(snake_out)
        self.ui.label_active_contours_input.setPixmap(QPixmap(qimg))
        
    def apply_active_contour(self):
           
        for i in range(len(self.my_points)):
            if i == 0:
                pass
            else:
                x0 = self.my_points[i-1][0]
                y0 = self.my_points[i-1][1]
                x1 = self.my_points[i][0]
                y1 = self.my_points[i][1]
        self.snake()
                
            
    def snake(self):
        image = cv2.imread(self.path_snake) 
        image = cv2.resize(image,(200,200), interpolation = cv2.INTER_AREA)
        img = np.copy(image)
        red = [0,0,255]
        points = np.array(self.my_points)
        x = points[:,0].reshape(len(self.my_points),1).astype(int)
        y = points[:,1].reshape(len(self.my_points),1).astype(int)
        points = np.concatenate((x,y),axis = 1)
        image_gray = cv2.imread(self.path_snake, 0) 
        image_gray = cv2.resize(image_gray,(200,200), interpolation = cv2.INTER_AREA)

        ## Detect edges using sobel 
        sobel_x_kernal = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        sobel_y_kernal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        ## Gaussian on image
        Gaussian_image = convolve2d(image_gray,self.gkern(1,5), mode='same')

        ## Apply sobel filter
        sobel_x = convolve2d(Gaussian_image, sobel_x_kernal, mode='same')
        sobel_y = convolve2d(Gaussian_image,sobel_y_kernal, mode='same')
        Gmag = (sobel_x**2) + (sobel_y**2)

        out_E = np.zeros((25))
        alpha = float(self.alpha.toPlainText())
        beta = float(self.beta.toPlainText())
        gama = float(self.gamma.toPlainText())
        copy_points = np.copy(points)
        iteration = int(self.iterations.toPlainText())

        for itration in range(iteration):
            for i in range(len(copy_points)):
                x , y = copy_points[i]
                if(i == len(copy_points)-1):
                    x_next , y_next = copy_points[0]
                    x_prev , y_prev = copy_points[i-1]
                else:
                    x_next , y_next = copy_points[i+1]
                    x_prev , y_prev = copy_points[i-1]

                window = self.create_window(2,x,y)
                for j in range(len(window)):
                    pix_x , pix_y = window[j] 
                    E_ex = -Gmag[pix_x,pix_y]
                    E_in_1 = ((x_next - pix_x)**2) + ((y_next - pix_y)**2)
                    E_in_2 = ((x_next - (2*pix_x) + x_prev)**2) + ((y_next - (2*pix_y) + y_prev)**2)
                    out_E[j] = (alpha * E_in_1) + (beta * E_in_2) + (gama * E_ex)

                index = np.where(out_E == out_E.min())[0][0]
                copy_points[i] = window[index] 
            image_out = np.copy(image)
            for point in copy_points:
                cv2.circle(image_out, (point[1],point[0]), 2, [200,0,0], -1)
        cv2.imwrite("imgs/snake.jpg", image_out)
        snake_out = cv2.imread('imgs/snake.jpg')
        qimg = qtimg.array2qimage(snake_out)
        self.ui.label_active_contours_output.setPixmap(QPixmap(qimg))
            
            
    def canny(self):
        image = cv2.imread(self.path_canny, 0) 
        sobel_x_kernal = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        sobel_y_kernal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Gauiss_img = self.conv(image,self.gkern(1,5))
        sobel_img_x = self.conv(Gauiss_img,sobel_x_kernal)
        sobel_img_y = self.conv(Gauiss_img,sobel_y_kernal)
        sobel_mag = np.sqrt((sobel_img_x**2) + (sobel_img_y**2))
        gradient_direction = np.arctan2(sobel_img_x, sobel_img_y)
        gradient_direction = np.rad2deg(gradient_direction)
        gradient_direction += 180
        new_image1 = self.non_max_suppression(sobel_mag, gradient_direction)
        weak = 50
        new_image2 = self.threshold(new_image1, 10, 70, weak=weak)
        new_image = self.hysteresis(new_image2, weak)
        self.ui.label_canny_output.show()
        self.ui.label_canny_output.setImage(np.rot90(new_image,1))
        
    def load_canny(self):
        self.browse(self.ui.label_canny_input)
        self.img_canny = cv2.imread(self.path_canny, 0) 
        self.ui.label_canny_input.show()
        self.ui.label_canny_input.setImage(np.rot90(self.img_canny, 1))
        
    def load_active_contour(self):
        self.img_countour = self.browse(self.ui.label_active_contours_input)
        qimg = qimage2ndarray.array2qimage(self.img_countour)
        self.ui.label_active_contours_input.setScaledContents(True)
        self.ui.label_active_contours_input.setPixmap(QPixmap(qimg))
        
    def load_hough(self):
        self.img_hough = self.browse(self.ui.label_hough_input)
        qimg = qimage2ndarray.array2qimage(self.img_hough)
        self.ui.label_hough_input.setScaledContents(True)
        self.ui.label_hough_input.setPixmap(QPixmap(qimg))
        
    def mouse_coordinates(self, e):
        c_x = int(e.pos().x() * len(self.img_countour[0])) / self.labelWidth
        c_y = int(e.pos().y() * len(self.img_countour)) / self.labelHeight
        self.my_points.append([c_x,c_y])


    def browse(self, label , RGB = False):
        file, _ = QFileDialog.getOpenFileName(self, "Browse", "Choose",
                                              "Filter -- All Files (*);;Image Files (*)")
        if(label == self.ui.label_hough_input):
            self.path_hough = file
            if file:
                self.imagenew = cv2.imread(file)
                self.imagenew = cv2.resize(self.imagenew,(200,200), interpolation = cv2.INTER_AREA)
                qimg = qtimg.array2qimage(self.imagenew)
                label.setPixmap(QPixmap(qimg))
                file_name = ntpath.basename(str(file))
                return self.imagenew
        elif(label == self.ui.label_active_contours_input):
            self.path_snake = file
            if file:
                self.imagenew = cv2.imread(file)
                self.imagenew = cv2.resize(self.imagenew,(200,200), interpolation = cv2.INTER_AREA)
                qimg = qtimg.array2qimage(self.imagenew)
                label.setPixmap(QPixmap(qimg))
                file_name = ntpath.basename(str(file))
                return self.imagenew
        elif(label == self.ui.label_canny_input):
            self.path_canny = file
            
        


    def show_error(self, message):
        errorBox = QMessageBox()
        errorBox.setIcon(QMessageBox.Warning)
        errorBox.setWindowTitle('WARNING')
        errorBox.setText(message)
        errorBox.setStandardButtons(QMessageBox.Ok)
        errorBox.exec_()
    
    def hough_lines_acc(self,img, rho_resolution=1, theta_resolution=1):
        ''' A function for creating a Hough Accumulator for lines in an image. '''
        height, width = img.shape # we need heigth and width to calculate the diag
        img_diagonal = np.ceil(np.sqrt(height**2 + width**2)) # a**2 + b**2 = c**2
        rhos = np.arange(-img_diagonal, img_diagonal + 1, rho_resolution)
        thetas = np.deg2rad(np.arange(-90, 90, theta_resolution))
        H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
        y_idxs, x_idxs = np.nonzero(img) # find all edge (nonzero) pixel indexes
        for i in range(len(x_idxs)): # cycle through edge points
            x = x_idxs[i]
            y = y_idxs[i]

            for j in range(len(thetas)): # cycle through thetas and calc rho
                rho = int((x * np.cos(thetas[j]) +
                           y * np.sin(thetas[j])) + img_diagonal)
                H[rho, j] += 1

        return H, rhos, thetas
    
    def hough_lines_draw(self,img, indicies, rhos, thetas):

        for i in range(len(indicies)):
            # reverse engineer lines from rhos and thetas
            rho = rhos[indicies[i][0]]
            theta = thetas[indicies[i][1]]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            # these are then scaled so that the lines go off the edges of the image
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def hough_simple_peaks(self,H, num_peaks):

        indices =  np.argpartition(H.flatten(), -2)[-num_peaks:]
        return np.vstack(np.unravel_index(indices, H.shape)).T
    def canny_edge_detector(self,input_image):
        input_pixels = input_image.load()
        width = input_image.width
        height = input_image.height

        # Transform the image to grayscale
        grayscaled = self.compute_grayscale(input_pixels, width, height)

        # Blur it to remove noise
        blurred = self.compute_blur(grayscaled, width, height)

        # Compute the gradient
        gradient, direction = self.compute_gradient(blurred, width, height)

        # Non-maximum suppression
        self.filter_out_non_maximum(gradient, direction, width, height)

        # Filter out some edges
        keep = self.filter_strong_edges(gradient, width, height, 20, 25)

        return keep


    def compute_grayscale(self,input_pixels, width, height):
        grayscale = np.empty((width, height))
        for x in range(width):
            for y in range(height):
                pixel = input_pixels[x, y]
                grayscale[x, y] = (pixel[0] + pixel[1] + pixel[2]) / 3
        return grayscale


    def compute_blur(self,input_pixels, width, height):
        # Keep coordinate inside image
        clip = lambda x, l, u: l if x < l else u if x > u else x

        # Gaussian kernel
        kernel = np.array([
            [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256],
            [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
            [6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256],
            [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
            [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256]
        ])

        # Middle of the kernel
        offset = len(kernel) // 2

        # Compute the blurred image
        blurred = np.empty((width, height))
        for x in range(width):
            for y in range(height):
                acc = 0
                for a in range(len(kernel)):
                    for b in range(len(kernel)):
                        xn = clip(x + a - offset, 0, width - 1)
                        yn = clip(y + b - offset, 0, height - 1)
                        acc += input_pixels[xn, yn] * kernel[a, b]
                blurred[x, y] = int(acc)
        return blurred


    def compute_gradient(self,input_pixels, width, height):
        gradient = np.zeros((width, height))
        direction = np.zeros((width, height))
        for x in range(width):
            for y in range(height):
                if 0 < x < width - 1 and 0 < y < height - 1:
                    magx = input_pixels[x + 1, y] - input_pixels[x - 1, y]
                    magy = input_pixels[x, y + 1] - input_pixels[x, y - 1]
                    gradient[x, y] = sqrt(magx**2 + magy**2)
                    direction[x, y] = atan2(magy, magx)
        return gradient, direction


    def filter_out_non_maximum(self,gradient, direction, width, height):
        for x in range(1, width - 1):
            for y in range(1, height - 1):
                angle = direction[x, y] if direction[x, y] >= 0 else direction[x, y] + pi
                rangle = round(angle / (pi / 4))
                mag = gradient[x, y]
                if ((rangle == 0 or rangle == 4) and (gradient[x - 1, y] > mag or gradient[x + 1, y] > mag)
                        or (rangle == 1 and (gradient[x - 1, y - 1] > mag or gradient[x + 1, y + 1] > mag))
                        or (rangle == 2 and (gradient[x, y - 1] > mag or gradient[x, y + 1] > mag))
                        or (rangle == 3 and (gradient[x + 1, y - 1] > mag or gradient[x - 1, y + 1] > mag))):
                    gradient[x, y] = 0


    def filter_strong_edges(self,gradient, width, height, low, high):
        # Keep strong edges
        keep = set()
        for x in range(width):
            for y in range(height):
                if gradient[x, y] > high:
                    keep.add((x, y))

        # Keep weak edges next to a pixel to keep
        lastiter = keep
        while lastiter:
            newkeep = set()
            for x, y in lastiter:
                for a, b in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                    if gradient[x + a, y + b] > low and (x+a, y+b) not in keep:
                        newkeep.add((x+a, y+b))
            keep.update(newkeep)
            lastiter = newkeep

        return list(keep)

    def hough_peaks(self,H, num_peaks, threshold=0, nhood_size=3):
        indicies = []
        H1 = np.copy(H)
        for i in range(num_peaks):
            idx = np.argmax(H1) # find argmax in flattened array
            H1_idx = np.unravel_index(idx, H1.shape) # remap to shape of H
            indicies.append(H1_idx)

            # surpess indicies in neighborhood
            idx_y, idx_x = H1_idx # first separate x, y indexes from argmax(H)
            # if idx_x is too close to the edges choose appropriate values
            if (idx_x - (nhood_size/2)) < 0: min_x = 0
            else: min_x = idx_x - (nhood_size/2)
            if ((idx_x + (nhood_size/2) + 1) > H.shape[1]): max_x = H.shape[1]
            else: max_x = idx_x + (nhood_size/2) + 1

            # if idx_y is too close to the edges choose appropriate values
            if (idx_y - (nhood_size/2)) < 0: min_y = 0
            else: min_y = idx_y - (nhood_size/2)
            if ((idx_y + (nhood_size/2) + 1) > H.shape[0]): max_y = H.shape[0]
            else: max_y = idx_y + (nhood_size/2) + 1

            # bound each index by the neighborhood size and set all values to 0
            for x in range(int(min_x), int(max_x)):
                for y in range(int(min_y), int(max_y)):
                    # remove neighborhoods in H1
                    H1[y, x] = 0

                    # highlight peaks in original H
                    if (x == min_x or x == (max_x - 1)):
                        H[y, x] = 255
                    if (y == min_y or y == (max_y - 1)):
                        H[y, x] = 255

        return indicies, H
    
    def non_max_suppression(self,gradient_magnitude, gradient_direction):
        image_row, image_col = gradient_magnitude.shape

        output = np.zeros(gradient_magnitude.shape)

        PI = 180

        for row in range(1, image_row - 1):
            for col in range(1, image_col - 1):
                direction = gradient_direction[row, col]

                if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                    before_pixel = gradient_magnitude[row, col - 1]
                    after_pixel = gradient_magnitude[row, col + 1]

                elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                    before_pixel = gradient_magnitude[row + 1, col - 1]
                    after_pixel = gradient_magnitude[row - 1, col + 1]

                elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                    before_pixel = gradient_magnitude[row - 1, col]
                    after_pixel = gradient_magnitude[row + 1, col]

                else:
                    before_pixel = gradient_magnitude[row - 1, col - 1]
                    after_pixel = gradient_magnitude[row + 1, col + 1]

                if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                    output[row, col] = gradient_magnitude[row, col]


        return output

    def threshold(self,image, low, high, weak):
        output = np.zeros(image.shape)

        strong = 255

        strong_row, strong_col = np.where(image >= high)
        weak_row, weak_col = np.where((image <= high) & (image >= low))

        output[strong_row, strong_col] = strong
        output[weak_row, weak_col] = weak

        return output

    def hysteresis(self,image, weak):
        image_row, image_col = image.shape

        top_to_bottom = image.copy()

        for row in range(1, image_row):
            for col in range(1, image_col):
                if top_to_bottom[row, col] == weak:
                    if top_to_bottom[row, col + 1] == 255 or top_to_bottom[row, col - 1] == 255 or top_to_bottom[row - 1, col] == 255 or top_to_bottom[
                        row + 1, col] == 255 or top_to_bottom[
                        row - 1, col - 1] == 255 or top_to_bottom[row + 1, col - 1] == 255 or top_to_bottom[row - 1, col + 1] == 255 or top_to_bottom[
                        row + 1, col + 1] == 255:
                        top_to_bottom[row, col] = 255
                    else:
                        top_to_bottom[row, col] = 0

        bottom_to_top = image.copy()

        for row in range(image_row - 1, 0, -1):
            for col in range(image_col - 1, 0, -1):
                if bottom_to_top[row, col] == weak:
                    if bottom_to_top[row, col + 1] == 255 or bottom_to_top[row, col - 1] == 255 or bottom_to_top[row - 1, col] == 255 or bottom_to_top[
                        row + 1, col] == 255 or bottom_to_top[
                        row - 1, col - 1] == 255 or bottom_to_top[row + 1, col - 1] == 255 or bottom_to_top[row - 1, col + 1] == 255 or bottom_to_top[
                        row + 1, col + 1] == 255:
                        bottom_to_top[row, col] = 255
                    else:
                        bottom_to_top[row, col] = 0

        right_to_left = image.copy()

        for row in range(1, image_row):
            for col in range(image_col - 1, 0, -1):
                if right_to_left[row, col] == weak:
                    if right_to_left[row, col + 1] == 255 or right_to_left[row, col - 1] == 255 or right_to_left[row - 1, col] == 255 or right_to_left[
                        row + 1, col] == 255 or right_to_left[
                        row - 1, col - 1] == 255 or right_to_left[row + 1, col - 1] == 255 or right_to_left[row - 1, col + 1] == 255 or right_to_left[
                        row + 1, col + 1] == 255:
                        right_to_left[row, col] = 255
                    else:
                        right_to_left[row, col] = 0

        left_to_right = image.copy()

        for row in range(image_row - 1, 0, -1):
            for col in range(1, image_col):
                if left_to_right[row, col] == weak:
                    if left_to_right[row, col + 1] == 255 or left_to_right[row, col - 1] == 255 or left_to_right[row - 1, col] == 255 or left_to_right[
                        row + 1, col] == 255 or left_to_right[
                        row - 1, col - 1] == 255 or left_to_right[row + 1, col - 1] == 255 or left_to_right[row - 1, col + 1] == 255 or left_to_right[
                        row + 1, col + 1] == 255:
                        left_to_right[row, col] = 255
                    else:
                        left_to_right[row, col] = 0

        final_image = top_to_bottom + bottom_to_top + right_to_left + left_to_right

        final_image[final_image > 255] = 255

        return final_image

    ## padding 
    def padding(self,f,image):
        n, m = image.shape
        pad = int((f-1)/2)
        padded_image = np.zeros((image.shape[0] + 2*pad, image.shape[1] + 2*pad ))
        padded_image[pad:padded_image.shape[0]-pad,pad:padded_image.shape[1]-pad ] = image

        return padded_image , pad 

    ## Convolution Matrix

    def conv(self,image,kernel):
        f = kernel.shape[0]
        kernel = np.flipud(np.fliplr(kernel))
        if(len(image.shape)==3):
            n, m, ch = image.shape
            out_img = np.zeros((n-f+(2*pad)+1, n-f+(2*pad)+1, 3))
            for k in range(ch):
                for j in range(n-f+1):
                    for i in range(n-f+1):
                        out_img[i,j,k] = (kernel*image_padded[i:i+f, j:j+f, k]).sum()
        elif(len(image.shape)==2):
            n, m = image.shape
            padded_image, pad = self.padding(f,image)
            out_img = np.zeros((n-f+(2*pad)+1, n-f+(2*pad)+1))
            for j in range(n-f+1):
                    for i in range(n-f+1):
                        out_img[i,j] = (kernel*padded_image[i:i+f, j:j+f]).sum()
        return out_img


    def show_gray(self,img1):
        fig = plt.figure(figsize=(10, 7))
        rows = 2
        columns = 1
        fig.add_subplot(rows, columns, 1)
        plt.imshow(img1.astype(np.uint8), cmap='gray')

    def create_window(self,size,x,y):
        window = []
        for i in range(-size,size+1):
            for j in range(-size,size+1):
                widow = window.append([x+i,y+j])
        return window

    def gkern(self,s,k):
        probs = [exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k,k+1)] 
        kernel = np.outer(probs, probs)
        return kernel
    
def main():
    app = QtWidgets.QApplication(sys.argv)
    application = App()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
