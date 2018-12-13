# Project: Search and Sample Return

Sergei Surovtsev 
Udacity Robotics Software Engineer Nanodegree
Class of November 2018

## Project Description

This project involves two basic stages of robot's operations: perception and actutation. We are given a simulator of Mars rover with task of mapping terrain and collecting yellow stones. Project is graded by performance metrics which are percent of mapped terrain and fidelity (accuracy of mapping).

## Technical Forumulation of Problem 

* Download] the simulator and take data in "Training Mode". Links are available at [Udacity project repository](https://github.com/udacity/RoboND-Rover-Project)
* Draft workflow on recorded test scenario on a [Jupyter Notebook](https://github.com/cwiz/RoboND-Rover-Project/blob/master/code/Rover_Project_Test_Notebook.ipynb)
* Modify code/perception.py and code/decision.py 

## Mathematical Models

### Perception

We are able to recieve following data from rover:

* Rover Camera Image (320x160 RGB)
* x, y coodriantes and yaw angle on a global map

We are given a task to map Rover's environment and locate positions of golden rocks. 

In order to map environment from a rover camera images we need to perform following steps:

1. Detect navigable ground, obstacles and yellow rocks on camera view
2. Perform perspective transform for top-down view
3. Fuse colelction of top-down images of navigable ground, obstacles and yellow stones into world map.

#### Perception 1: Image Segmentation

![techincal-vision](https://github.com/cwiz/RoboND-Rover-Project/blob/master/report_images/technical-vision.png?raw=true "Technical Vision")

I'm using basic techniques to segment image such as image blurring and color thresholding. This works fine for given environment, but won't work well under different light conditions, time of day or weather. A real image segmentation is usully performed using Deep Learning Semantic Segmentation.

##### Perception 1: Image Segmentation 1: Navigable Ground

![segmentation-ground](https://github.com/cwiz/RoboND-Rover-Project/blob/master/report_images/segmentation-ground.png?raw=true "Ground Segmentation")

RGB color thresholding is used here with values of (150, 150, 150). Then image is blured with 10x10 mask.

```python
def detect_ground(image):
    ground = color_threshold_rgb(image, (150, 150, 150))
    return cv2.blur(ground, (10,10))
```

##### Perception 1: Image Segmentation 2: Obstacles

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

##### Perception 1: Image Segmentation 3: Yellow Stones

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

#### Perception 2: Perspective Transform

We use OpenCV's warpPerspective to transform camera images into a top-down view. This approach requires either calibration images or mathematical model to compose a transformation matrix. We use calibration setting supplied in Jupyter Notebook.

#### Perception 3: Conversion to Global Coordinates

![global-coordinates](https://github.com/cwiz/RoboND-Rover-Project/blob/master/report_images/global-coordinates.png?raw=true "Global Coordinates")

In order to transform set of top-down image into a world map I've taken following steps:

1. Convert top-down image in such a way that router vision axis is parallel to x axis and originating from center of y axis
2. Create a vision mask that clips areas of top-down view that are located far from rover (bacause perceptive transform if reliable for areas close to rover). Otherwise too much noise would be on an area map.
3. Transform image from rover-centric to world-frame coodinate system using rover's global x, y positions and yaw angle.

### Autonomous Mapping 

I've written a WorldMapper class in Jupyter Notebook. It's is given a test data record and is required to perform mapping and asses fidelity metric. The core of mapping procedure is in process_step function

```python
    def process_step(self, index):
        # Load raw image
        img = mpimg.imread(self.images[index])
        # IMU data from mapping
        x, y, yaw = self.xpos[index], self.ypos[index], self.yaw[index]
        
        # Process raw frame to get maps for Ground, Yellow Stone and Obstacles
        o = detect_obstacles(img)
        g = detect_ground(img)
        ys = detect_yellow_stone(img)

        # Perform perspective transform
        wg = perspect_transform(g, source, destination)
        wys = perspect_transform(ys, source, destination)
        wo = perspect_transform(o, source, destination)

        # Perform rotation
        ag = adjust_warped(wg) * self.mask
        ays = adjust_warped(wys) * self.mask
        ao = adjust_warped(wo) * self.mask
        
        # Technical Vision       
        tv = colorize(g, 100) + colorize(ys, 255) + colorize(o, 50)
        
        # Warped View
        wv = colorize(ag, 100) + colorize(ays, 255) + colorize(ao, 50)
        wv = cv2.resize(wv, (img.shape[1], img.shape[0]))
        
        # Ground Map
        wmg = convert_from_rover_frame_to_world_frame(ag, x, y, yaw)
        wmys = convert_from_rover_frame_to_world_frame(ays, x, y, yaw)
        wmo = convert_from_rover_frame_to_world_frame(ao, x, y, yaw)
        
        # map building version
        self.ground_map = np.logical_or(wmg, self.ground_map)
        self.obstacle_map = np.logical_or(wmo, self.obstacle_map)
        self.yellow_stone_map = np.logical_or(wmys, self.obstacle_map)
        
        # last frame 
        self.lf_ground_map = wmg
        self.lf_obstacle_map = wmo
        self.lf_yellow_stone_map = wmys
        
        return (img, tv, wv, x, y, yaw)
```

render_movie() builds frameset for mapping progress and render_movie_frame() renders a random fram to test the algorithm.

### Actuation

### Results

This project requires 75% accuracy and 60% fidelity metrics.

![results](https://github.com/cwiz/RoboND-Rover-Project/blob/master/output/minimum_requirements.PNG?raw=true "Results")

Test run [video](https://www.youtube.com/watch?v=NWho2wcnQFc).