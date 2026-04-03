import math
import skimage
import numpy as np
import matplotlib.pyplot as plt

from save_dicom import save_as_dicom

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
height, width = img_grey.shape

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
        
        rows = np.array(rows)
        cols = np.array(cols)
        
        mask = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        pixels = img_grey[rows[mask], cols[mask]]

        if len(pixels) > 0:
            sinogram[view, D] = np.mean(pixels) #srednia z jasnosci

    current_angle += delta_a

def reconstruct(sinogram, width, height, n, phi, delta_a, r):
    # 1. Create a blank canvas for the image and a counter for normalization
    reconstruction = np.zeros((height, width))
    hits = np.zeros((height, width))
    
    offset_x = width / 2
    offset_y = height / 2
    current_angle = 0
    
    # 2. Loop through every view (angle) in the sinogram
    for view in range(len(sinogram)):
        current_angle_rad = math.radians(current_angle)
        
        # Emitter position (xe, ye)
        xe = r * math.cos(current_angle_rad)
        ye = r * math.sin(current_angle_rad)
        
        # 3. For every detector at this angle
        for D in range(n):
            # Get the value recorded in the sinogram
            val = sinogram[view, D]
            
            # Use YOUR original detectorCords function
            xd, yd = detectorCords(current_angle_rad, n, phi, r, D)
            
            # Use YOUR original get_pixels_on_line function
            rows, cols = get_pixels_on_line(xe + offset_x, ye + offset_y, 
                                            xd + offset_x, yd + offset_y)
            
            # Ensure we only work with pixels inside the image boundaries
            mask = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
            r_mask = rows[mask]
            c_mask = cols[mask]
            
            # 4. Backproject: Add the sinogram value to the image path
            reconstruction[r_mask, c_mask] += val
            hits[r_mask, c_mask] += 1
            
        current_angle += delta_a

    # 5. Normalization: Divide by hits to fix the "bright center" effect
    # We use np.where to avoid dividing by zero if a pixel was never hit
    reconstruction = np.divide(reconstruction, hits, out=np.zeros_like(reconstruction), where=hits!=0)
    
    # Final step: scale the values to 0-1 range for clear plotting
    # if np.max(reconstruction) > 0:
    #     reconstruction = (reconstruction - np.min(reconstruction)) / (np.max(reconstruction) - np.min(reconstruction))
        
    v_min, v_max = np.percentile(reconstruction, (5, 99))
    rescaled = skimage.exposure.rescale_intensity(reconstruction, in_range=(v_min, v_max))
    rescaled = np.clip(rescaled, 0, None)
    return rescaled

# Execution
reconstructed = reconstruct(sinogram, width, height, n, phi, delta_a, r)



# Display results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Sinogram")
plt.imshow(sinogram, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Reconstructed Image (Backprojection)")
plt.imshow(reconstructed, cmap='gray')
plt.show()