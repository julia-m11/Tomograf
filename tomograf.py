import math
import skimage
import numpy as np
import matplotlib.pyplot as plt

# wejscie : krok delta a, n - liczba detektorow, phi - rozpietosc katowa miedzy detektorami
# wyjscie : xd, yd - wspolrzedne detektora d

# a - obecna lokalizacja detektora, i - numer detektora, phi - rozpietosc katowa miedzy detektorami, r - odleglosc od zrodla do detektora
def detectorCords(a,n,phi,r,i):

    if i == 0:
        xd = r * math.cos(a + math.pi - phi/2)
        yd = r * math.sin(a + math.pi - phi/2)
    elif i == n-1:
        xd = r * math.cos(a + math.pi + phi/2)
        yd = r * math.sin(a + math.pi + phi/2)
    else: #di
        xd = r * math.cos(a + math.pi - phi/2 + i * phi/(n-1))
        yd = r * math.sin(a + math.pi - phi/2 + i * phi/(n-1))

    return xd, yd

def get_pixels_on_line(x1, y1, x2, y2):
    p1 = [int(round(y1)), int(round(x1))]
    p2 = [int(round(y2)), int(round(x2))]
    
    return skimage.draw.line_nd(p1, p2, endpoint=True, integer=True) # zwraca dwie tablice - jedna z wierszami, druga z kolumnami

img = skimage.io.imread("tomograf-obrazy/Kropka.jpg")
img_grey = skimage.color.rgb2gray(img)
img_grey = (img_grey * 255).astype(np.uint8)
width, height = img_grey.shape

# ------ to podawane na wejsciu przez użytkownika --------
delta_a = 1.0 #rozmiar kroku
n = 180 #liczba detektorów
phi = math.pi / 2 # rozpiętość układu w stopbniach
# -----------------------------------------------------
r = math.sqrt(width**2 + height**2) / 2 + 10 

angles = np.arange(0, 360, delta_a) #lista wszystkich kątów detektorów
num_steps = len(angles)
sinogram = np.zeros((num_steps, n)) # liczba skoków kąta x liczba detektorów
offset_x = width / 2
offset_y = height / 2

current_angle = 0
for view in range(len(sinogram)):
    current_angle_rad = math.radians(current_angle)
    xe = r * math.cos(current_angle_rad) #xe,ye - pozycja emitera
    ye = r * math.sin(current_angle_rad)
    for D in range(n):
        xd, yd = detectorCords(current_angle_rad, n, phi, r, D)
        rows, cols = get_pixels_on_line(xe + offset_x, ye + offset_y, 
                                        xd + offset_x, yd + offset_y)
        
        mask = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        pixels = img_grey[rows[mask], cols[mask]]

        if len(pixels) > 0:
            sinogram[view, D] = np.mean(pixels) #srednia z jasnosci

    current_angle += delta_a

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1) 
plt.imshow(img_grey, cmap='gray', aspect='equal')

plt.subplot(1, 2, 2)  
plt.imshow(sinogram, cmap='gray') 
plt.show()