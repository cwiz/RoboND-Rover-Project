# Project: Search and Sample Return

**Sergei Surovtsev**
<br/>
Udacity Robotics Software Engineer Nanodegree
<br/>
Class of November 2018

## Project Description

This project involves two basic stages of robot's operations: perception and decision making. We are given a simulator of Mars rover with task of mapping terrain and collecting yellow stones. Project is graded by performance metrics which are percent of mapped terrain and fidelity (accuracy of mapping).

## Technical Formulation of Problem 

* Download the simulator and take data in "Training Mode". Links are available at [Udacity project repository](https://github.com/udacity/RoboND-Rover-Project)
* Draft workflow on recorded test scenario on a [Jupyter Notebook](https://github.com/cwiz/RoboND-Rover-Project/blob/master/code/Rover_Project_Test_Notebook.ipynb)
* Modify code/perception.py and code/decision.py 

## Mathematical Models

### Perception

Following data is available from a Rover:

* Camera image (320x160 RGB)
* x, y coordinates and yaw angle on a global map

The task is to map Rover's environment and locate positions of golden rocks. 

In order to map environment from a rover camera images we need to perform following steps:

1. Detect navigable ground, obstacles and yellow rocks on camera view
2. Perform perspective transform for top-down view
3. Fuse collection of top-down images of navigable ground, obstacles and yellow stones into world map.

#### Image Segmentation

![techincal-vision](https://github.com/cwiz/RoboND-Rover-Project/blob/master/report_images/technical-vision.png?raw=true "Technical Vision")

I'm using basic techniques to segment image such as image blurring and color thresholding. This works fine for given environment, but won't work well under different light conditions, time of day or weather. A real image segmentation is usually performed using Deep Learning Semantic Segmentation.

##### Navigable Ground Segmentation

![segmentation-ground](https://github.com/cwiz/RoboND-Rover-Project/blob/master/report_images/segmentation-ground.png?raw=true "Ground Segmentation")

RGB color thresholding is used here with values of (150, 150, 150). Then image is blurred with 10x10 mask.

```python
def detect_ground(image):
    ground = color_threshold_rgb(image, (150, 150, 150))
    return cv2.blur(ground, (10,10))
```

##### Obstacles Segmentation

![segmentation-obstacles](https://github.com/cwiz/RoboND-Rover-Project/blob/master/report_images/segmentation-obstacles.png?raw=true "Obstacle Segmentation")

Image is converted into grayscale, blurred with 10x10 mask and then binary segmented with value of 80.

```python
def detect_obstacles(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.blur(grayscale,(10,10))

    grayscale[grayscale <  80] = 1
    grayscale[grayscale != 1]  = 0 
    
    return grayscale
```

##### Yellow Stones Segmentation

![segmentation-rock](https://github.com/cwiz/RoboND-Rover-Project/blob/master/report_images/segmentation-rock.png?raw=true "Rock Segmentation")

Image is blurred with 2x2 mask and then HSL segmentation is used to find areas with yellow color. Theshold values used here are: (0-45, 50-200, 130-255). HSL is chosen here instead of RGB because of more convenient representation of yellow color.

```python
def detect_yellow_stone(image):
    yellow_stone = cv2.blur(image,(2,2))
    return color_limit_hsl(yellow_stone, hsl_lower=[0,50,130])

def color_limit_hsl(image, hsl_lower=[20,120,80], hsl_upper=[45, 200, 255]):
    # convert image to hls colour space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)

    # hls thresholding for yellow
    lower = np.array(hsl_lower,dtype = "uint8")
    upper = np.array(hsl_upper,dtype = "uint8")
    mask = cv2.inRange(hls, lower, upper)
    
    return mask 
```

#### Perspective Transform

OpenCV's warpPerspective transforms camera images into a top-down view images. This approach requires either calibration images or mathematical model to compose a transformation matrix. 

#### Conversion to Global Coordinates

![global-coordinates](https://github.com/cwiz/RoboND-Rover-Project/blob/master/report_images/global-coordinates.png?raw=true "Global Coordinates")

In order to transform set of top-down image into a world map I've taken following steps:

1. Convert top-down image in such a way that router vision axis is parallel to x axis and originating from center of y axis
2. Create a vision mask that clips areas of top-down view that are located far from Rover.
3. Transform image from rover-centric to world-frame coordinates system using rover's global x, y positions and yaw angle.

#### Rover Perception

I've written a ```WorldMapper``` class in Jupyter Notebook to test that mapping scenario. The core of mapping procedure is in ```WorldMapper.process_step()``` function. ```WorldMapper.render_movie()``` builds frameset for mapping progress and ```WorldMapper.render_movie_frame()``` renders a random frame to test the algorithm.

Rover executes following code on perception phase. We feed rover's actuation algorithm with list of angles on which transition is possible. To do that we extract angles in ```extract_angles_and_distances()```.

```python
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
    
    # Perspective transformation is only valid on small roll and pitch angles
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
```

### Decision Making

This revision of project uses code supplied in ```code/decision.py```. It is a decision tree that uses following data to generate rover commands:

* List of traversible angles
* Rover's technical limitation such as max speed and angles of transition
* Current phase of operation plan.

A modification of this algorithm might include introduction of new phases: mapping and sample collection, building a traversible graph for sample collection. However minimum required metrics for submission are met without modification of decision code.

### Results

This project requires 75% accuracy and 60% fidelity metrics.

![results](https://github.com/cwiz/RoboND-Rover-Project/blob/master/output/minimum_requirements.PNG?raw=true "Results")

Test run [video](https://www.youtube.com/watch?v=NWho2wcnQFc).