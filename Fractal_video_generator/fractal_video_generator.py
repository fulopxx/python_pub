import numpy as np
import matplotlib.pyplot as plt
import imageio
from numba import cuda, vectorize, int64, float32
import traceback
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.cm as cm
import torch

'''
#profiling
import cProfile
'''

class FractalGui(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title('Fractal Generator')
        self.geometry('1800x1800')  # slightly adjust window size to accommodate side by side layout

        # Video parameters
        self.n_frames = 20 # The number of frames in the video
        self.width = 640 # The width of each frame
        self.height = 480 # The height of each frame
        self.zoom_speed = 0.12 # The zoom speed for the fractal
        self.fps = 30 # The frames per second for the video
        self.max_iter = 256 # Max iteration

        # Status
        self.status = tk.StringVar()
        self.status.set("Status: Ready")

        # Colormap options
        self.colormaps = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
        ]
        self.selected_colormap = tk.StringVar()
        self.selected_colormap.set(self.colormaps[0])  # set default value
        self.selected_colormap.trace('w', self.update_initial_frame)  # Add callback for variable change

        # Control Frame
        control_frame = tk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # Canvas
        self.canvas = tk.Canvas(self, width=self.width, height=self.height)
        self.canvas.pack(side=tk.LEFT)

        self.option_menu = tk.OptionMenu(control_frame, self.selected_colormap, *self.colormaps)
        self.option_menu.pack(side=tk.LEFT)

        self.start_button = tk.Button(control_frame, text="Start Render", command=self.start_render)
        self.start_button.pack(side=tk.LEFT)

        self.status_label = tk.Label(control_frame, textvariable=self.status)
        self.status_label.pack(side=tk.LEFT)

        # Render the first frame
        self.initial_frame()

    def start_render(self):
        self.status.set("Status: Rendering...")
        self.update_idletasks()
        create_video(self)  # We only pass 'self' here

    def update_image(self, img):
        tk_image = ImageTk.PhotoImage(Image.fromarray(img))
        self.canvas.create_image(0, 0, anchor='nw', image=tk_image)
        self.canvas.image = tk_image

    '''
    def initial_frame(self):
        fig, ax = plt.subplots()
        zoom_level = 1.0
        x_center = -0.75  # x-coordinate of the center of the zoom
        y_center = 0.0  # y-coordinate of the center of the zoom
        draw(x_center - zoom_level, x_center + zoom_level, y_center - zoom_level, y_center + zoom_level, self.width, self.height,
             self.max_iter, ax)
        ax.axis('off')
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(
            fig.canvas.get_width_height()[::-1] + (3,))
        self.update_image(image)
        plt.close(fig)
    '''

    #the original code
    def initial_frame(self):
        fig, ax = plt.subplots()
        zoom_level = 1.0
        x_center = -0.75  # x-coordinate of the center of the zoom
        y_center = 0.0  # y-coordinate of the center of the zoom
        draw(x_center - zoom_level, x_center + zoom_level, y_center - zoom_level, y_center + zoom_level, self.width, self.height,
             self.max_iter, ax, self.selected_colormap.get())
        ax.axis('off')
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(
            fig.canvas.get_width_height()[::-1] + (3,))
        self.update_image(image)
        plt.close(fig)


    def update_initial_frame(self, *args):  # New function to handle colormap change
        self.initial_frame()


#original code
@vectorize(['int64(complex64, int32)'], target='cuda')
def mandelbrot(c, max_iter):
    z = c
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter


'''
#this is the fixed (v4) Distance Estimation coloring technique CUDA based (torch)
# Define the function for calculating the iteration count.
def mandelbrot_iter(c, max_iter):
    z = torch.zeros(c.shape, dtype=torch.cfloat, device=c.device)
    output = torch.zeros(c.shape, dtype=torch.int, device=c.device)
    for i in range(max_iter):
        notdone = torch.less(z.real*z.real + z.imag*z.imag, 4.0)
        output[notdone] = i
        z[notdone] = z[notdone]**2 + c[notdone]
    output[output == max_iter - 1] = 0
    return output
'''

'''
#this is the fixed (v3) Distance Estimation coloring technique CPU BASED (NO CUDA)
# Define the function for calculating the iteration count.
def mandelbrot_iter(c, max_iter):
    z = np.zeros(c.shape, dtype=complex)
    output = np.zeros(c.shape, dtype=int)
    for i in range(max_iter):
        notdone = np.less(z.real*z.real + z.imag*z.imag, 4.0)
        output[notdone] = i
        z[notdone] = z[notdone]**2 + c[notdone]
    output[output == max_iter - 1] = 0
    return output
'''

'''
#this is the fixed (v3) Distance Estimation coloring technique
# Define the function for calculating the iteration count.
@vectorize(['int64(complex64, int64)'], target='cuda')
def mandelbrot_iter(c, max_iter):
    z = c
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter
'''

'''
#this is the fixed (v2) Distance Estimation coloring technique
# Define the function for calculating the iteration count.
@cuda.vectorize(['int64(complex64, int64)'], target='cuda')
def mandelbrot_iter(c, max_iter):
    z = c
    for n in range(max_iter):
        if abs(z) > 2:
            return n   # Return the iteration count
        z = z*z + c
    return max_iter  # Return the maximum iteration count if no divergence
'''

'''
#this is the fixed (v4) Distance Estimation coloring technique CUDA based (torch)
# Define the function for calculating the final distance.
def mandelbrot_dist(c, max_iter):
    z = torch.zeros(c.shape, dtype=torch.cfloat, device=c.device)
    output = torch.zeros(c.shape, dtype=torch.float, device=c.device)
    mask = torch.ones(c.shape, dtype=torch.bool, device=c.device)

    for i in range(max_iter):
        z[mask] = z[mask] * z[mask] + c[mask]
        mask[torch.abs(z) > 2] = False
        output[~mask] = torch.abs(z[~mask])
    output[mask] = torch.abs(z[mask])

    return output
'''

'''
#this is the fixed (v3) Distance Estimation coloring technique CPU BASED (NO CUDA)
# Define the function for calculating the final distance.
def mandelbrot_dist(c, max_iter):
    z = np.zeros(c.shape, dtype=complex)
    output = np.zeros(c.shape, dtype=float)
    mask = np.ones(c.shape, dtype=bool)

    for i in range(max_iter):
        z[mask] = z[mask] * z[mask] + c[mask]
        mask[np.abs(z) > 2] = False
        output[~mask] = np.abs(z[~mask])
    output[mask] = np.abs(z[mask])

    return output
'''

'''
#this is the fixed (v3) Distance Estimation coloring technique
# Define the function for calculating the final distance.
@vectorize(['float32(complex64, int64)'], target='cuda')
def mandelbrot_dist(c, max_iter):
    z = c
    for n in range(max_iter):
        if abs(z) > 2:
            return abs(z)   # Return the distance
        z = z*z + c
    return abs(z)  # Return the final distance
'''

'''
#this is the fixed (v2) Distance Estimation coloring technique
# Define the function for calculating the final distance.
@cuda.vectorize(['float32(complex64, int64)'], target='cuda')
def mandelbrot_dist(c, max_iter):
    z = c
    for n in range(max_iter):
        if abs(z) > 2:
            return abs(z)   # Return the distance
        z = z*z + c
    return abs(z)  # Return the final distance
'''

#this is the Distance Estimation coloring technique
def color_map_distance(distance, max_distance):
    normalized_distance = distance / max_distance
    colormap = torch.tensor(cm.get_cmap('plasma')(normalized_distance.cpu().numpy()), device='cuda')
    return colormap



#original function
def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    n3 = np.empty((height, width), dtype=np.int64)

    # Generate the grid of complex numbers
    c = r1 + r2[:, None]*1j

    # Reshape to 1-dimensional array
    c_1d = c.ravel()

    # Run the mandelbrot function
    mandelbrot(c_1d, max_iter, out=n3.ravel())  # Call mandelbrot on the 1D array

    return (r1, r2, n3)


'''
# this would replace your mandelbrot_set function
def mandelbrot_torch(xmin,xmax,ymin,ymax,width,height,max_iter):
    r1 = torch.linspace(xmin, xmax, width, device='cuda')
    r2 = torch.linspace(ymin, ymax, height, device='cuda')
    c = r1 + r2[:, None]*1j
    return mandelbrot(c, max_iter)
'''

'''
# alternative CUDA torch based
def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    r1 = torch.linspace(xmin, xmax, width, device='cuda')
    r2 = torch.linspace(ymin, ymax, height, device='cuda')
    c = r1 + r2[:, None] * 1j
    c = c.type(torch.complex64)
    n3 = mandelbrot_iter(c, max_iter)
    d3 = mandelbrot_dist(c, max_iter)
    return (n3, d3)
'''

'''
#this is the Distance Estimation coloring technique fixed to CUDA based (torch)
def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    # Create torch tensors instead of numpy arrays
    r1 = torch.linspace(xmin, xmax, width, dtype=torch.float32).cuda()
    r2 = torch.linspace(ymin, ymax, height, dtype=torch.float32).cuda()

    # Create complex tensor in PyTorch
    c = torch.complex(r1, r2[:, None])
    n3 = mandelbrot_iter(c, max_iter)
    d3 = mandelbrot_dist(c, max_iter)

    return (n3.cpu(), d3.cpu())  # Move the result back to CPU
'''

'''
#this is the Distance Estimation coloring technique fixed to CPU based (NOCUDA) version
def mandelbrot_set(xmin,xmax,ymin,ymax,width,height,max_iter):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    c = r1 + r2[:,None]*1j
    c = c.astype(np.complex64)
    n3 = mandelbrot_iter(c, max_iter)
    d3 = mandelbrot_dist(c, max_iter)
    return (n3, d3)
'''

'''
#this is the Distance Estimation coloring technique
def mandelbrot_set(xmin,xmax,ymin,ymax,width,height,max_iter):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    n3 = cuda.to_device(np.empty((width,height), dtype = np.int64))
    d3 = cuda.to_device(np.empty((width,height), dtype = np.float32))
    c = r1 + r2[:,None]*1j
    c = c.astype(np.complex64)
    cuda.synchronize()
    mandelbrot_iter(c, max_iter, out=n3)
    mandelbrot_dist(c, max_iter, out=d3)
    cuda.synchronize()
    return (n3.copy_to_host(), d3.copy_to_host())
'''


#original code

def draw(xmin, xmax, ymin, ymax, width, height, max_iter, ax, colormap):
    d = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)
    ax.imshow(d[2], extent=(xmin, xmax, ymin, ymax), cmap=colormap)


'''
#this is the fixed (v4) Distance Estimation coloring technique fixed to CUDA based (torch)
def draw(xmin, xmax, ymin, ymax, width, height, max_iter, ax):
    iteration_counts, distances = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)
    colors = color_map_distance(distances, torch.max(distances))
    ax.imshow(colors.cpu().numpy(), extent=(xmin, xmax, ymin, ymax))
'''

'''
#this is the fixed (v3) Distance Estimation coloring technique
def draw(xmin, xmax, ymin, ymax, width, height, max_iter, ax):
    iteration_counts, distances = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)
    colors = color_map_distance(distances, np.max(distances))
    ax.imshow(colors, extent=(xmin, xmax, ymin, ymax))
'''

'''
#this is the fixed (v2) Distance Estimation coloring technique
def draw(xmin, xmax, ymin, ymax, width, height, max_iter, ax):
    d = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)
    colors = color_map_distance(d[3], np.max(d[3]))  # Use the distances returned by mandelbrot_set
    ax.imshow(colors, extent=(xmin, xmax, ymin, ymax))
'''


#original function code
def generate_frames(gui):
    frames = []
    fig, ax = plt.subplots()

    x_center = 0.10150891662818479
    y_center = 0.6330939302717072

    for i in range(gui.n_frames):
        print(f"Generating frame {i+1} of {gui.n_frames}")
        gui.status.set("Status: Generating frame " + str(i+1) + " of " + str(gui.n_frames) + "...")
        ax.clear()
        zoom_level = np.exp(-gui.zoom_speed * i)
        draw(x_center - zoom_level, x_center + zoom_level, y_center - zoom_level, y_center + zoom_level, gui.width, gui.height, gui.max_iter, ax, gui.selected_colormap.get())
        ax.axis('off') # to get rid of the axes for the video
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

        # Display image in GUI
        gui.update_image(image)
        gui.update_idletasks()
        gui.update()

    plt.close(fig)
    return frames

'''
#this is the Distance Estimation coloring technique
def generate_frames(gui):
    frames = []
    fig, ax = plt.subplots()

    x_center = 0.10150891662818479
    y_center = 0.6330939302717072

    for i in range(gui.n_frames):
        print(f"Generating frame {i+1} of {gui.n_frames}")
        gui.status.set("Status: Generating frame " + str(i+1) + " of " + str(gui.n_frames) + "...")
        ax.clear()
        zoom_level = np.exp(-gui.zoom_speed * i)
        draw(x_center - zoom_level, x_center + zoom_level, y_center - zoom_level, y_center + zoom_level, gui.width, gui.height, gui.max_iter, ax)
        ax.axis('off') # to get rid of the axes for the video
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

        # Display image in GUI
        gui.update_image(image)  # This will now handle images created using distance estimation
        gui.update_idletasks()
        gui.update()

    plt.close(fig)
    return frames
'''

def create_video(gui):
    try:
        print("Starting to generate frames.")
        frames = generate_frames(gui)
        print("Finished generating frames. Now creating video.")
        imageio.mimsave('fractal_zoom.mp4', frames, fps=gui.fps, macro_block_size=None, codec='libx264', quality=10, ffmpeg_params=['-pix_fmt', 'yuv420p'], output_params=['-s', f'{gui.width}x{gui.height}'])
        #imageio.mimsave('fractal_zoom.mp4', frames, fps=gui.fps)
        print("Video saved as fractal_zoom.mp4.")
        gui.status.set("Status: Video saved as fractal_zoom.mp4.")
    except Exception as e:
        gui.status.set(f"An error occurred: {e}")
        print("Here is the traceback:")
        traceback.print_exc()

'''
#for profiling
def main():
    app = FractalGui()
    app.mainloop()

if __name__ == '__main__':
    import cProfile
    cProfile.run('main()', filename='output.prof')
'''

gui = FractalGui()
gui.mainloop()