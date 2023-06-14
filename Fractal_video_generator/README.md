## Fractal Video Generator

It is a simple MANDELBROT set ZOOM application using tkinter as GUI, matplotlib, imageio, numba, numpy, torch - CPU and/or CUDA acceleration implementations (commmented out the corresponding part of the code in this revision)

### USING

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
