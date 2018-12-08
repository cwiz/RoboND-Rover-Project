import numpy as np
import cv2


# Color Manipulation

def color_limit_hsl(image, hsl_lower=[20,120,80], hsl_upper=[45, 200, 255]):
    # convert image to hls colour space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)

    # hls thresholding for yellow
    lower = np.array(hsl_lower,dtype = "uint8")
    upper = np.array(hsl_upper,dtype = "uint8")
    mask = cv2.inRange(hls, lower, upper)
    
    return mask 

def color_threshold_rgb(img, rgb_thresh=(160, 160, 180)):
    color_select = np.zeros_like(img[:,:,0])
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    color_select[above_thresh] = 1
    return color_select

# Object Detection

def detect_ground(image):
    ground = color_threshold_rgb(image, (150, 150, 150))
    return cv2.blur(ground, (10,10))

def detect_yellow_stone(image):
    yellow_stone = cv2.blur(image,(2,2))
    return color_limit_hsl(yellow_stone, hsl_lower=[0,50,130])

def detect_obstacles(image):
    obstacles = color_threshold_rgb(image, rgb_thresh=(85, 85, 85))
    obstacles = np.invert(obstacles)

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.blur(grayscale,(10,10))

    grayscale[grayscale <  80] = 1
    grayscale[grayscale != 1]  = 0 
    
    return grayscale

# Perspective Manipulation and Translation

def rotate_pix(xpix, ypix, yaw):
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))     
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad)) 
    return xpix_rotated, ypix_rotated

def perspect_transform(img, src, dst):   
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    return warped

# Rover Visibility Mask
def vision_mask():
    center = np.array([150, 0])
    ones = np.ones((320, 160))
    for i in range(ones.shape[0]):
        for j in range(ones.shape[1]):
            point = np.array([i, j])
            distance = np.linalg.norm(point-center)
            if distance != 0:
                ones[i, j] = 1/(distance)
            if distance >= 40:
                ones[i, j] = 0
    ones[ones!=0] = 1
    return ones

def adjust_warped(warped):
    return np.flip(warped.T)

def convert_from_rover_frame_to_world_frame(img, x, y, yaw, world_size=200):
    result = np.zeros((world_size, world_size, 1))
    scaled = cv2.resize(img, None, fx=0.2, fy=0.2)
    
    for i in range(scaled.shape[0]):
        for j in range(scaled.shape[1]):
            (_i, _j) = rotate_pix(j, i-32, yaw)
            _x = int(_i + x)
            _y = int(_j + y)
            
            if 0 <= _x < result.shape[0] and 0 <= _y < result.shape[1]:
                result[_x, _y] = scaled[i, j]
    
    return result

# Rover perception step
def perception_step(Rover):
    o = detect_obstacles(img)
    g = detect_ground(img)
    ys = detect_yellow_stone(img)

    # Technical Vision       
    Rover.vision_image[:, :, 0] = o
    Rover.vision_image[:, :, 1] = ys
    Rover.vision_image[:, :, 2] = g

    # Perform perspective transform
    wg = perspect_transform(g, source, destination)
    wys = perspect_transform(ys, source, destination)
    wo = perspect_transform(o, source, destination)

    # Perform rotation
    mask = vision_mask()
    ag = adjust_warped(wg) * mask
    ays = adjust_warped(wys) * mask
    ao = adjust_warped(wo) * mask
    
    # Ground Map
    wmg = convert_from_rover_frame_to_world_frame(ag, x, y, yaw)
    wmys = convert_from_rover_frame_to_world_frame(ays, x, y, yaw)
    wmo = convert_from_rover_frame_to_world_frame(ao, x, y, yaw)
    
    Rover.worldmap[:,:,0] += wmo
    Rover.worldmap[:,:,1] += wmys
    Rover.worldmap[:,:,2] += wmg

    return Rover
