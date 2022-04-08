import cv2
import dlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
from scipy import interpolate
from pylab import *
from skimage import color
from imutils import face_utils
import imutils
import xlrd2
import statistics
from statistics import mean
import streamlit as st

#global variables
global pil_image
global imOrg

#function to pick style and access rgb values
def style_picker(n):
    loc = ('./StyleColour.xlsx')
    wb = xlrd2.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    rgbvals=[]
    for i in range(1,13):
        rgbvals.append(int(sheet.cell_value(n, i)))
    return rgbvals

def run_script(image, makeup_number_choice):
    global pil_image
    global imOrg
    
    #print(type(image))
    #img import and convert to array
    image = Image.open(image)
    image = image.convert("RGB")
    image = np.array(image)
    #image = np.array(Image.open('./harshu.jpeg'))

    #preprocess image
    image = imutils.resize(image, width=500)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #face detection and landmarks detection library load
    hog_face_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    #face detection
    faces = hog_face_detector(gray, 1)
    shape = predictor(gray, faces[0])

    #extract facial features
    shape = face_utils.shape_to_np(shape)
    pil_image = Image.fromarray(image)

    #draw on image caller
    d = ImageDraw.Draw(pil_image, 'RGBA')

    #take input styles
    # print("Enter style")
    # print("1. Minimal")
    # print("2. Formal")
    # print("3. Party ")
    # print("4. Goth")

    #Taking input on type of colour
    #m=int(input())
    m = makeup_number_choice

    rvals=style_picker(m)

    #facedata in list form
    shape = shape.tolist()
    for i,j in enumerate(shape):
        shape[i] = (j[0], j[1])

    # store indices of landmark points[0-67] to be used
    indices = [48,49,50,51,52,53,54,64,63,62,61,60,48]
    top_lip = [shape[i] for i in indices]
    indices = [48,60,67,66,65,64,54,55,56,57,58,59,48]
    bottom_lip = [shape[i] for i in indices]
    indices = [36,37,38,39,40,41,36]
    left_eye = [shape[i] for i in indices]
    indices = [42,43,44,45,46,47,42]
    right_eye = [shape[i] for i in indices]
    indices=[17,18,19,20,21,17]
    left_eyebrow=[shape[i] for i in indices]
    indices=[22,23,24,25,26,22]
    right_eyebrow=[shape[i] for i in indices]

    # Draw lipstick, eyeliner and eyebrow pencil on image
    d.polygon(top_lip, fill=(rvals[0],rvals[1],rvals[2],100))
    d.polygon(bottom_lip, fill=(rvals[0],rvals[1],rvals[2],100))
    d.line(left_eye, fill=(rvals[3],rvals[4],rvals[5], 150), width=3)
    d.line(right_eye, fill=(rvals[3],rvals[4],rvals[5], 150), width=3)
    d.polygon(left_eyebrow, fill=(rvals[6],rvals[7],rvals[8], 150))
    d.polygon(right_eyebrow, fill=(rvals[6],rvals[7],rvals[8], 150))

    #show image with lipstick and eyeliner and eyebrow pencil
    plt.imshow(pil_image)

    #blush colour
    Rg, Gg, Bg = (rvals[9],rvals[10],rvals[11])

    #load image with initial makeup
    pil_image = np.asarray(pil_image)

    #preprocess for blush
    height, width = pil_image.shape[:2]
    intensity = 0.5
    imOrg = pil_image.copy()

    def get_boundary_points(x, y):
        # obtain all the points between given vertice of polygon
        tck, u = interpolate.splprep([x, y], s=0, per=1)
        unew = np.linspace(u.min(), u.max(), 1000)
        xnew, ynew = interpolate.splev(unew, tck, der=0)
        tup = np.c_[xnew.astype(int), ynew.astype(int)].tolist()
        coord = list(set(tuple(map(tuple, tup))))
        coord = np.array([list(elem) for elem in coord])
        return np.array(coord[:, 0], dtype=np.int32), np.array(coord[:, 1], dtype=np.int32)

    def get_interior_points(x, y):
        intx = []
        inty = []
        
        # interior function to extend the list with points within
        # given range i.e. a to b

        def ext(a, b, i):
            a, b = round(a), round(b)
            intx.extend(np.arange(a, b, 1).tolist())
            inty.extend((np.ones(b - a) * i).tolist())

        x, y = np.array(x), np.array(y)
        xmin, xmax = np.amin(x), np.amax(x)
        xrang = np.arange(xmin, xmax + 1, 1)
        for i in xrang:
            ylist = y[np.where(x == i)]
            ext(np.amin(ylist), np.amax(ylist), i)
        return np.array(intx, dtype=np.int32), np.array(inty, dtype=np.int32)

    def apply_blush_color(r=Rg, g=Gg, b=Bg):
        global pil_image

        # normalize and change the intensities of pixels in 'LAB' color scheme
        val = color.rgb2lab((pil_image / 255.)).reshape(width * height, 3)
        L, A, B = statistics.mean(val[:, 0]), statistics.mean(val[:, 1]), statistics.mean(val[:, 2])
        L1, A1, B1 = color.rgb2lab(np.array((r / 255., g / 255., b / 255.)).reshape(1, 1, 3)).reshape(3, )
        ll, aa, bb = (L1 - L) * intensity, (A1 - A) * intensity, (B1 - B) * intensity
        val[:, 0] = np.clip(val[:, 0] + ll, 0, 100)
        val[:, 1] = np.clip(val[:, 1] + aa, -127, 128)
        val[:, 2] = np.clip(val[:, 2] + bb, -127, 128)

        # change the image array back to 'RGB' scheme
        pil_image = color.lab2rgb(val.reshape(height, width, 3)) * 255

    def smoothen_blush(x, y):
        global imOrg
        imgBase = np.zeros((height, width))

        
        # Fill the shape of blush with color
        # c_() from pylab just zips two arrays index wise
        
        cv2.fillConvexPoly(imgBase, np.array(np.c_[x, y], dtype='int32'), 1)
        # Blur the colour using GaussianBlur function
        imgMask = cv2.GaussianBlur(imgBase, (51, 51), 0)
        imgBlur3D = np.ndarray([height, width, 3], dtype='float')
        imgBlur3D[:, :, 0] = imgMask
        imgBlur3D[:, :, 1] = imgMask
        imgBlur3D[:, :, 2] = imgMask

        # Transform the image by adding blurred patch on cheek
        imOrg = (imgBlur3D * pil_image + (1 - imgBlur3D) * imOrg).astype('uint8')

    indices = [1,2,3,4,48,31,36]
    left_cheek_x = [shape[i][0] for i in indices]
    left_cheek_y = [shape[i][1] for i in indices]

    # Get the pixel points within polygon and apply color on those points
    left_cheek_x, left_cheek_y = get_boundary_points(left_cheek_x, left_cheek_y)
    left_cheek_y, left_cheek_x = get_interior_points(left_cheek_x, left_cheek_y)
    apply_blush_color()
    smoothen_blush(left_cheek_x, left_cheek_y)

    #plt.imshow(imOrg)
    indices = [15,14,13,12,54,35,45]
    right_cheek_x = [shape[i][0] for i in indices]
    right_cheek_y = [shape[i][1] for i in indices]

    # Get the pixel points within polygon and apply color on those points
    right_cheek_x, right_cheek_y = get_boundary_points(right_cheek_x, right_cheek_y)
    right_cheek_y, right_cheek_x = get_interior_points(right_cheek_x, right_cheek_y)
    apply_blush_color()
    smoothen_blush(right_cheek_x, right_cheek_y)

    plt.imshow(imOrg)
    #plt.show()
    plt.savefig('applied_result.png', dpi=1000)
    return plt


st.title("Makeup app")
image = st.file_uploader("Upload an image")
if(image):
    st.image(image)
makeup_choice = st.radio("Choose makeup style", ["Minimal","Formal","Party","Goth"])

makeup_number_map = {
    "Minimal" : 1,
    "Formal" : 2,
    "Party" : 3,
    "Goth": 4 
}

makeup_number_choice = makeup_number_map[makeup_choice]

clicked = st.button("Run")
if(clicked):
    run_script(image, makeup_number_choice)
    st.image("./applied_result.png")


