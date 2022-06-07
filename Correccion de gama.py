import cv2 
import HSI
import numpy as np 

img = cv2.imread('IMG.jpg')
cv2.imshow("Original Image", img)
cv2.waitKey(0)

###################### Escala de grises ##############

img = cv2.imread('IMG.jpg')
B = img[:,:,0]
G = img[:,:,1]
R = img[:,:,2]

BGR = 0.299*(B)+0.587*(G)+0.114*(R)
BGR = BGR.astype(np.uint8)

cv2.imshow('Escala de grises',BGR)
cv2.waitKey(0)


#############              CMYK             #########################

#cv2 is used for OpenCV library
image = cv2.imread('IMG.jpg')

#imread is use to read an image from a location

img = image.astype(np.float64)/255.
K = 1 - np.max(img, axis=2)
C = (1-img[...,2] - K)/(1-K)
M = (1-img[...,1] - K)/(1-K)
Y = (1-img[...,0] - K)/(1-K)

CMYK_image= (np.dstack((C,M,Y,K)) * 255).astype(np.uint8)


cv2.imshow("CMYK Image", CMYK_image)
cv2.waitKey(0)


#############              CMY             #########################

img = cv2.imread('IMG.jpg')
B = img[:,:,0]
G = img[:,:,1]
R = img[:,:,2]

CMY = (1-B)+(1-G)+(1-R)
CMY = CMY.astype(np.uint8)
cv2.imshow('CMY',CMY)
cv2.waitKey(0)

#############              HSI             #########################

img = cv2.imread('IMG.jpg')
hsi = HSI.RGB_TO_HSI(img)
# Display HSV Image
cv2.imshow('HSI', hsi)
# The three value channels
cv2.imshow('HSI H Channel', hsi[:, :, 0])
cv2.imshow('HSI S Channel', hsi[:, :, 1])
cv2.imshow('HSI I Channel', hsi[:, :, 2])
cv2.waitKey(0)

#############              HSV            #########################


def HSV():
    img = cv2.imread('IMG.jpg')
    a = np.float_(img)
    m,n,l=a.shape
    H = np.zeros((m,n),np.float_)
    S = np.zeros((m,n), np.float_)
    V = np.zeros((m,n), np.float_)
    r,g,b = cv2.split(img)
    r, g, b = r/255.0, g/255.0, b/255.0

    for y in range(n):
        for x in range(m):
            mx = max((b[x, y], g[x, y], r[x, y]))
            mn = min((b[x, y], g[x, y], r[x, y]))
            dt=mx-mn

            if mx == mn:
                H[x, y] = 0
            elif mx == r[x, y]:
                if g[x, y] >= b[x, y]:
                    H[x, y] = (60 * ((g[x, y]) - b[x, y]) / dt)
                else:
                    H[x, y] = (60 * ((g[x, y]) - b[x, y]) / dt)+360
            elif mx == g[x, y]:
                H[x, y] = 60 * ((b[x, y]) - r[x, y]) / dt + 120
            elif mx == b[x, y]:
                H[x, y] = 60 * ((r[x, y]) - g[x, y]) / dt+ 240
            H[x,y] =int( H[x,y] / 2)

            #S
            if mx == 0:
                S[x, y] = 0
            else:
                S[x, y] =int( dt/mx*255)
            #V
            V[x, y] =int( mx*255)



    H = np.uint8(H)
    S = np.uint8(S)
    V = np.uint8(V)
    hsv = cv2.merge([H,S,V]) 
    hsv=np.array(hsv,dtype='uint8')
    cv2.imshow('HSV', hsv)
    cv2.waitKey(0)


#############                YCbCr          #########################

def YCrCb():
    a = cv2.imread('IMG.jpg')
    a_f = np.float_(a)
    m,n,l=a.shape
    Y1=np.zeros((m,n,l),np.uint8)
    cr=np.zeros((m,n,l),dtype = np.float_)
    cb=np.zeros((m,n,l),dtype = np.float_)
   
    for y in range(n):
        for x in range(m):    
            Y1[x][y] = 0.144*a_f[x][y][0] + 0.587*a_f[x][y][1] + 0.299*a_f[x][y][2]
    
 
    for y in range(n):
        for x in range(m):    
            valor = -0.1687*a_f[x][y][2] - 0.3313*a_f[x][y][1] + 0.5*a_f[x][y][0] +128
            if valor > 0:
                cb[x][y] = valor
            
            else :
                cb[x][y] = 0
    
    for y in range(n):
        for x in range(m):            
            valor2 = 0.5*a_f[x][y][2] - 0.4187 *a_f[x][y][1] - 0.0813*a_f[x][y][0] + 128

            if valor2 > 0:
                cr[x][y] = valor2
            
            else :
                cr[x][y] = 0
  
    Y1 = np.uint8(Y1)
    cb = np.uint8(cb)
    cr = np.uint8(cr)

    C1 = Y1 [:,:,0]
    C2 = cr [:,:,1]
    C3 = cb [:,:,2]
    y = cv2.merge([C1,C2,C3])
    cv2.imshow('YCbCr', cv2.merge([C1,C2,C3]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run():
    a = cv2.imread('IMG.jpg')
    a_f = np.float_(a)
    m,n,l=a.shape
    HSV()
    YCrCb()

if __name__=="__main__":
    run()