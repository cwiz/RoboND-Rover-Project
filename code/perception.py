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
    yellow_stone = cv2.blur(image,(5,5))
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


dst_size = 5 
bottom_offset = 6
src = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
dst = np.float32([[320/2 - dst_size, 160 - bottom_offset],
    [320/2 + dst_size, 160 - bottom_offset],
    [320/2 + dst_size, 160 - 2*dst_size - bottom_offset], 
    [320/2 - dst_size, 160 - 2*dst_size - bottom_offset],
    ])   
M = cv2.getPerspectiveTransform(src, dst)

def perspect_transform(img):
    warped = cv2.warpPerspective(img, M, (320, 160))
    return warped

# Rover Visibility Mask
def vision_mask(ds=60):
    center = np.array([150, 0])
    ones = np.ones((320, 160))
    for i in range(ones.shape[0]):
        for j in range(ones.shape[1]):
            point = np.array([i, j])
            distance = np.linalg.norm(point-center)
            if distance != 0:
                ones[i, j] = 1/(distance)
            if distance >= ds:
                ones[i, j] = 0
    ones[ones!=0] = 1
    return ones

def adjust_warped(warped):
    return np.flip(warped.T)

def convert_from_rover_frame_to_world_frame(img, x, y, yaw, world_size=200):
    result = np.zeros((world_size+76*2, world_size+76*2))
    # Scale
    scaled = cv2.resize(img, None, fx=0.2, fy=0.2)
    scaled_2x = np.zeros((scaled.shape[0]+12, scaled.shape[1]*2+12))
    # Scale 2x amnd move to center
    scaled_2x[6:scaled.shape[0]+6, scaled.shape[1]+6:-6] = scaled
    scaled_2x = np.flip(scaled_2x, axis=0)
    # Rotate around center
    rows,cols = scaled_2x.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),yaw-90,1)
    dst = cv2.warpAffine(scaled_2x,M,(cols,rows))
    # overlay
    _x = int(x + dst.shape[0]/2)
    _y = int(y + dst.shape[1]/2)
    result[_x:_x+dst.shape[0],_y:_y+dst.shape[1]] = dst

    adjusted = result[76:-76,76:-76]
    
    adjusted[adjusted!=0] = 1
    adjusted[int(x), int(y)] = 1

    return adjusted

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def extract_angles_and_distances(binary_img):
    angles = []
    ypos, xpos = binary_img.nonzero()
    for i in range(len(ypos)):
        point = np.array([ypos[i]-160, xpos[i]])
        unit = unit_vector(point)
        angle = np.arctan(unit[0]/unit[1])
        angles.append(angle)

    return np.array(angles)
    
# Rover perception step
mask = vision_mask(60)
obstacle_mask = vision_mask(35)
def perception_step(Rover):
    img = Rover.img
    o = detect_obstacles(img)
    g = detect_ground(img)
    ys = detect_yellow_stone(img)

    # Perform perspective transform
    wg = perspect_transform(g)
    wys = perspect_transform(ys)
    wo = perspect_transform(o)

    # Perform rotation
    ag = adjust_warped(wg) * mask
    ays = adjust_warped(wys) * mask
    ao = adjust_warped(wo) * mask

    # Technical Vision       
    Rover.vision_image[:, :, 0] = adjust_warped(ao) * 255
    Rover.vision_image[:, :, 1] = adjust_warped(ays) * 255
    Rover.vision_image[:, :, 2] = adjust_warped(ag) * 255

    # Ground Map
    x, y, yaw = Rover.pos[0], Rover.pos[1], Rover.yaw

    wmg = convert_from_rover_frame_to_world_frame(ag, x, y, yaw)
    wmys = convert_from_rover_frame_to_world_frame(ays, x, y, yaw)
    wmo = convert_from_rover_frame_to_world_frame(ao, x, y, yaw)
    
    if np.abs(np.sin(np.deg2rad(Rover.roll))) <= 0.05 and np.abs(np.sin(np.deg2rad(Rover.pitch))) <= 0.05:

        wm = Rover.worldmap.T

        wm[0,:,:] += wmo
        wm[1,:,:] += wmys
        wm[2,:,:] += wmg

        Rover.worldmap = wm.T

    # Basic navigation
    Rover.nav_angles = extract_angles_and_distances(ag)

    # Perform rotation
    cv2.putText(Rover.vision_image,"yaw: %f" % (np.rad2deg(np.mean(Rover.nav_angles))), (20, 20), 
    cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)

    return Rover
