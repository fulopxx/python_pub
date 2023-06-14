## Fractal Video Generator

It is a simple MANDELBROT set ZOOM application using tkinter as GUI, matplotlib, imageio, numba, numpy, torch - CPU and/or CUDA acceleration implementations (commmented out the corresponding part of the code in this revision)

### Requirements
- I used Python 3.9 interpreter, PyPi packages
- installed CUDA 11.8 (and tested on NVIDIA GeForce RTX 2070 8GB)
- torch needs to prebuilt with using the corresponding CUDA version (description on the torch website) myversion: Version: 2.1.0.dev20230525+cu118

### USING

GUI has 2 buttons:
- colorspace dropdown menu, where user can select predefined colosspaces applied to the mandelbrot set, it has impact on the rendered output frame colorspace range
- Start Render button which will start rendering the video, and when finishes then it will export an h264 video 'fractal_zoom.mp4' into  the source folder
- also has a preview canvas in the gui, and the first rendered frame can be viewed when selecting the colorspace before starting the rendering

configuring the exported and rendered frames should be done in the FractalGui Class:
```
        # Video parameters
        self.n_frames = 20 # The number of frames in the video
        self.width = 640 # The width of each frame
        self.height = 480 # The height of each frame
        self.zoom_speed = 0.12 # The zoom speed for the fractal
        self.fps = 30 # The frames per second for the video
        self.max_iter = 256 # Max iteration
```

There is two option for rendering:
- normal
- Distance Estimation coloring technique CUDA based (torch) (v4 is the latest working) anda CPU based version also (slower) ->Distance Estimation coloring is a more advanced coloring scheme but slower rendering time
