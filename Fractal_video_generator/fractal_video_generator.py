from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import matplotlib.pyplot as plt
import imageio
from numba import cuda, vectorize, int64, float32
import traceback
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.cm as cm
import torch

'''
#profiling
import cProfile
'''

ZOOM_TARGETS = {
    "Original": (0.10150891662818479, 0.6330939302717072),
    "Seahorse Valley": (-0.74364386269, 0.13182590271),
    "Double Spiral Valley": (-0.77568377, 0.13646737),
}

RENDER_RESOLUTIONS = {
    "Quick test (160 x 90)": (160, 90),
    "Tiny (256 x 144)": (256, 144),
    "Small (320 x 180)": (320, 180),
    "Draft (480 x 270)": (480, 270),
    "360p (640 x 360)": (640, 360),
    "450p (800 x 450)": (800, 450),
    "540p (960 x 540)": (960, 540),
    "630p (1120 x 630)": (1120, 630),
    "HD (1280 x 720)": (1280, 720),
    "Full HD (1920 x 1080)": (1920, 1080),
    "QHD (2560 x 1440)": (2560, 1440),
    "4K UHD (3840 x 2160)": (3840, 2160),
}
DEFAULT_RESOLUTION = "Full HD (1920 x 1080)"
FRAME_RATES = (24, 30, 60)
DEFAULT_DURATION = "3"
DEFAULT_FPS = 30
DEFAULT_ZOOM_LOG = 0.12 * (3 * DEFAULT_FPS - 1)


def duration_to_frames(duration_text, fps):
    if fps not in FRAME_RATES:
        raise ValueError("Choose 24, 30, or 60 fps.")
    try:
        duration = Decimal(duration_text.strip().replace(",", "."))
    except InvalidOperation as error:
        raise ValueError("Enter a duration between 0.5 and 60 seconds.") from error
    if not duration.is_finite() or not Decimal("0.5") <= duration <= Decimal("60"):
        raise ValueError("Enter a duration between 0.5 and 60 seconds.")
    return int((duration * fps).to_integral_value(rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class RenderSettings:
    width: int
    height: int
    fps: int
    frame_count: int
    center: tuple[float, float]
    colormap: str
    max_iter: int
    zoom_log: float = DEFAULT_ZOOM_LOG


class FractalGui(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title('Fractal Generator')
        self.geometry('1800x1800')  # slightly adjust window size to accommodate side by side layout

        # The existing startup preview is independent of the output resolution.
        self.preview_width, self.preview_height = RENDER_RESOLUTIONS[DEFAULT_RESOLUTION]
        self.max_iter = 256
        self._rendering = False

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

        # Keep the target and its coordinates together on a separate control row.
        zoom_frame = ttk.Frame(self)
        zoom_frame.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)
        ttk.Label(zoom_frame, text="Zoom target:").pack(side=tk.LEFT, padx=(0, 5))
        self.selected_zoom_target = tk.StringVar(value="Original")
        self.zoom_target_menu = ttk.Combobox(
            zoom_frame, textvariable=self.selected_zoom_target,
            values=tuple(ZOOM_TARGETS), state="readonly", width=22)
        self.zoom_target_menu.pack(side=tk.LEFT)
        self.zoom_target_menu.bind("<<ComboboxSelected>>", self.update_zoom_target)

        self.zoom_coordinates = tk.StringVar()
        self._zoom_coords_font = tkfont.nametofont("TkDefaultFont").copy()
        default_size = self._zoom_coords_font.cget("size")
        smaller_size = max(1, abs(default_size) - 2)
        self._zoom_coords_font.configure(
            size=smaller_size if default_size >= 0 else -smaller_size)
        ttk.Label(zoom_frame, textvariable=self.zoom_coordinates,
                  font=self._zoom_coords_font).pack(side=tk.LEFT, padx=(8, 0))

        settings_frame = ttk.Frame(self)
        settings_frame.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)
        self.selected_resolution = tk.StringVar(value=DEFAULT_RESOLUTION)
        self.duration_seconds = tk.StringVar(value=DEFAULT_DURATION)
        self.selected_fps = tk.StringVar(value=str(DEFAULT_FPS))
        self.render_summary = tk.StringVar()

        ttk.Label(settings_frame, text="Resolution:").pack(side=tk.LEFT)
        self.resolution_menu = ttk.Combobox(
            settings_frame, textvariable=self.selected_resolution,
            values=tuple(RENDER_RESOLUTIONS), state="readonly", width=23)
        self.resolution_menu.pack(side=tk.LEFT, padx=(4, 8))

        ttk.Label(settings_frame, text="Duration (s):").pack(side=tk.LEFT)
        self.duration_entry = ttk.Entry(
            settings_frame, textvariable=self.duration_seconds, width=7)
        self.duration_entry.pack(side=tk.LEFT, padx=(4, 8))

        ttk.Label(settings_frame, text="FPS:").pack(side=tk.LEFT)
        self.fps_menu = ttk.Combobox(
            settings_frame, textvariable=self.selected_fps,
            values=FRAME_RATES, state="readonly", width=4)
        self.fps_menu.pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(self, textvariable=self.render_summary).pack(
            side=tk.TOP, anchor="w", padx=4, pady=2)
        for variable in (self.selected_resolution, self.duration_seconds, self.selected_fps):
            variable.trace_add("write", self.update_render_summary)
        self.update_render_summary()

        # Canvas
        self.canvas = tk.Canvas(self, width=self.preview_width, height=self.preview_height,
                                borderwidth=0, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT)
        self._preview_source = None
        self._preview_photo = None
        self._preview_item = self.canvas.create_image(0, 0, anchor=tk.CENTER)
        self.canvas.bind("<Configure>", self._redraw_preview)

        self.option_menu = tk.OptionMenu(control_frame, self.selected_colormap, *self.colormaps)
        self.option_menu.pack(side=tk.LEFT)

        self.start_button = tk.Button(control_frame, text="Start Render", command=self.start_render)
        self.start_button.pack(side=tk.LEFT)

        self.status_label = tk.Label(control_frame, textvariable=self.status)
        self.status_label.pack(side=tk.LEFT)

        # Render the first frame
        self.update_zoom_target()

    def start_render(self):
        if self._rendering:
            return
        try:
            settings = self.get_render_settings()
        except ValueError as error:
            self.status.set(f"Status: {error}")
            return

        self._rendering = True
        self.status.set("Status: Rendering...")
        self.set_render_controls_enabled(False)
        try:
            self.update_idletasks()
            create_video(self, settings)
        finally:
            self._rendering = False
            try:
                self.set_render_controls_enabled(True)
            except tk.TclError:
                # The window may have been closed while rendering.
                pass

    def set_render_controls_enabled(self, enabled):
        ttk_state = ["!disabled"] if enabled else ["disabled"]
        for widget in (self.zoom_target_menu, self.resolution_menu,
                       self.duration_entry, self.fps_menu):
            widget.state(ttk_state)
        tk_state = "normal" if enabled else "disabled"
        self.option_menu.configure(state=tk_state)
        self.start_button.configure(state=tk_state)

    def get_render_settings(self):
        width, height = RENDER_RESOLUTIONS[self.selected_resolution.get()]
        fps = int(self.selected_fps.get())
        frame_count = duration_to_frames(self.duration_seconds.get(), fps)
        return RenderSettings(width, height, fps, frame_count,
                              self.get_zoom_center(), self.selected_colormap.get(),
                              self.max_iter)

    def update_render_summary(self, *args):
        try:
            settings = self.get_render_settings()
        except ValueError as error:
            self.render_summary.set(str(error))
            return
        duration = settings.frame_count / settings.fps
        summary = (
            f"{settings.width} x {settings.height} | {settings.frame_count} frames "
            f"at {settings.fps} fps | {duration:.2f} s")
        if final_grid_has_duplicates(settings):
            summary += " | Final zoom: float32 grid repeats some coordinates"
        self.render_summary.set(summary)

    def get_zoom_center(self):
        return ZOOM_TARGETS[self.selected_zoom_target.get()]

    def update_zoom_target(self, event=None):
        x_center, y_center = self.get_zoom_center()
        self.zoom_coordinates.set(f"x = {x_center!r}, y = {y_center!r}")
        self.update_render_summary()
        self.initial_frame()

    def update_image(self, img):
        self._preview_source = Image.fromarray(img)
        self._redraw_preview()

    def _redraw_preview(self, event=None):
        if self._preview_source is None:
            return

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width <= 1 or height <= 1:
            # The first frame can arrive before Tk has laid out the canvas.
            return

        preview = self._preview_source
        if preview.width > width or preview.height > height:
            preview = preview.copy()
            preview.thumbnail((width, height), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(preview, master=self.canvas)
        self.canvas.coords(self._preview_item, width // 2, height // 2)
        self.canvas.itemconfigure(self._preview_item, image=photo)
        self._preview_photo = photo

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
        x_center, y_center = self.get_zoom_center()
        draw(x_center - zoom_level, x_center + zoom_level,
             y_center - zoom_level, y_center + zoom_level,
             self.preview_width, self.preview_height,
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


def frame_bounds(settings, index):
    # Keep the same endpoint when duration or FPS changes.
    progress = index / max(1, settings.frame_count - 1)
    half_height = np.exp(-settings.zoom_log * progress)
    half_width = half_height * settings.width / settings.height
    x, y = settings.center
    return x - half_width, x + half_width, y - half_height, y + half_height


def final_grid_has_duplicates(settings):
    xmin, xmax, ymin, ymax = frame_bounds(settings, settings.frame_count - 1)
    x = np.linspace(xmin, xmax, settings.width, dtype=np.float32)
    y = np.linspace(ymin, ymax, settings.height, dtype=np.float32)
    return bool(np.any(x[1:] == x[:-1]) or np.any(y[1:] == y[:-1]))


def generate_frames(settings):
    palette = plt.get_cmap(settings.colormap)
    levels = np.arange(settings.max_iter + 1, dtype=np.float64)
    for index in range(settings.frame_count):
        _, _, counts = mandelbrot_set(
            *frame_bounds(settings, index), settings.width, settings.height, settings.max_iter)
        low, high = int(counts.min()), int(counts.max())
        # Normalize a small lookup table, not a full-resolution float image.
        normalized = (
            np.clip((levels - low) / (high - low), 0, 1)
            if high > low else np.zeros_like(levels)
        )
        rgb = palette(normalized, bytes=True)[:, :3]
        yield rgb[counts]

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

def create_video(gui, settings):
    try:
        output = Path("fractal_zoom.mp4")
        # Keep the previous output intact if calculation or encoding fails.
        with TemporaryDirectory(prefix=".fractal-render-", dir=output.resolve().parent) as temp_dir:
            partial = Path(temp_dir) / output.name
            with imageio.get_writer(
                    partial, format="FFMPEG", mode="I", fps=settings.fps,
                    macro_block_size=None, codec="libx264", quality=10,
                    pixelformat="yuv420p") as writer:
                for index, frame in enumerate(generate_frames(settings), start=1):
                    message = f"Generating frame {index} of {settings.frame_count}"
                    print(message)
                    gui.status.set(f"Status: {message}...")
                    writer.append_data(frame)
                    gui.update_image(frame)
                    gui.update()
            if not partial.is_file() or partial.stat().st_size == 0:
                raise RuntimeError("FFmpeg did not produce an output video.")
            partial.replace(output)

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
