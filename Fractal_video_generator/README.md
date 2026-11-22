# Fractal Video Generator

A Windows-tested Python desktop application for rendering Mandelbrot zooms to H.264 MP4. The Tkinter interface lets you choose a zoom target, colormap, resolution, duration, and frame rate. Preview the start, middle, and end of the zoom and see an approximate compute time before exporting. The renderer uses Numba and CUDA on an NVIDIA GPU.

[Earlier interface screenshot](GUI.jpg) · [Sample video](fractal_zoom.mp4)

## Requirements

- 64-bit Windows and CPython 3.9 with Tcl/Tk.
- An NVIDIA GPU, a compatible NVIDIA driver and **CUDA Toolkit 12.1**. This setup was tested on a GeForce RTX 2070 (8 GB). `nvcc --version` reports the installed toolkit version; the CUDA version shown by `nvidia-smi` is driver information and does not confirm toolkit installation.
- The Windows packages in [`requirements.txt`](requirements.txt). The pinned PyTorch wheel is Windows x64 and Python 3.9 specific and is about 2.5 GB to download. The script imports PyTorch, but **Numba, not PyTorch, computes the fractal**. The ImageIO FFmpeg package supplies the encoder for the tested setup; no separate FFmpeg installation was needed.
- The NVIDIA CUDA Python binding enabled with `NUMBA_CUDA_USE_NVIDIA_BINDING=1`. The requirements include `cuda-python` and its Windows dependencies. On the tested machine, switching colormaps with Numba's default binding caused a CUDA context error and could leave the process running after the window closed. The NVIDIA binding passed both checks.

The pinned CUDA and package versions describe the tested setup, not a minimum supported configuration. CPU-only rendering and the experimental distance-estimation variants are not selectable features in this revision.

## Set up and run

Run the following PowerShell commands **from the `Fractal_video_generator` directory**, where `fractal_video_generator.py` and `requirements.txt` are located. For an existing installation, reuse its Python 3.9 virtual environment instead of creating a second one.

```powershell
$project = (Get-Location).Path
$venv = Join-Path $env:LOCALAPPDATA 'venvs\fractal-video-generator-py39'
$py = Join-Path $venv 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $py)) {
    py -3.9 -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python 3.9 environment.' }
}

& $py -m pip install --only-binary=:all: -r (Join-Path $project 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $py -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
```

Launch with a separate working directory so the generated video does not replace the repository's sample:

```powershell
$out = Join-Path $project 'video-output'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$video = Join-Path $out 'fractal_zoom.mp4'
if (Test-Path -LiteralPath $video) {
    throw 'An output video already exists. Move it or choose another working directory first.'
}

$previousBinding = $env:NUMBA_CUDA_USE_NVIDIA_BINDING
Push-Location -LiteralPath $out
try {
    $env:NUMBA_CUDA_USE_NVIDIA_BINDING = '1'
    & $py (Join-Path $project 'fractal_video_generator.py')
} finally {
    $env:NUMBA_CUDA_USE_NVIDIA_BINDING = $previousBinding
    Pop-Location
}
```

### PyCharm

Create or edit a Python run configuration for `fractal_video_generator.py`. Select the project's dedicated Python 3.9 virtual environment, set **Working directory** to the existing `video-output` folder, and add `NUMBA_CUDA_USE_NVIDIA_BINDING=1` under **Environment variables**. Keep any existing variables, such as `PYTHONUNBUFFERED=1`. PyCharm is optional; the PowerShell launch above uses the same project settings.

## Choose settings, preview, and export

1. Choose a **Zoom target**: `Original` (default), `Seahorse Valley`, or `Double Spiral Valley`. The exact `x` and `y` coordinates appear next to the menu, and the startup view updates to the selected target. Choose a colormap, such as `plasma`, using the separate colormap menu.
2. Set **Resolution**, **Duration (s)** and **FPS**. Duration accepts **0.5 to 60 seconds**, with a decimal point or comma. The frame rate can be **24, 30, or 60 fps**. The status line shows the frame count and actual duration after rounding to a whole number of frames. Defaults are **1920 × 1080, 3 seconds, 30 fps** (90 frames).
3. Click **Preview** to inspect three **320 × 180** frames labeled **Start**, **Middle**, and **End**. They use the same zoom path and framing as export, but not its pixel detail. The images are kept in RAM; this button does not invoke FFmpeg or write a video.
4. Click **Start Render** to export. Wait for the save message and open `video-output\fractal_zoom.mp4`. The program always writes into the current working directory. It streams native-size RGB frames to FFmpeg rather than storing the entire clip in memory.

The startup image and on-screen previews stay centered and fit the canvas without stretching or cropping, with upscaling limited to 2×. Resizing the window does not recompute the fractal or change the export. The single startup image uses the selected center but a different framing path from the three-image preview. Use **Preview** to judge the export composition.

The checked-in [`fractal_zoom.mp4`](fractal_zoom.mp4) is a sample, not an input. Move an output you want to keep before starting another export in the same directory. On a successful render, the application can replace the previous output. The PowerShell launch check only applies when launching from the example above, not to repeated renders in the app or launches from PyCharm.

### Output resolutions

| Preset | Resolution |
| --- | ---: |
| Quick test | 160 × 90 |
| Tiny | 256 × 144 |
| Small | 320 × 180 |
| Draft | 480 × 270 |
| 360p | 640 × 360 |
| 450p | 800 × 450 |
| 540p | 960 × 540 |
| 630p | 1120 × 630 |
| HD | 1280 × 720 |
| Full HD (default) | 1920 × 1080 |
| QHD | 2560 × 1440 |
| 4K UHD | 3840 × 2160 |

Use **Quick test** to check the export pipeline. Its low pixel count is not intended for judging fine detail. All presets are 16:9. Changing duration or FPS changes how many frames describe the zoom, not its endpoint. Maximum iterations remain fixed at 256 and the initial colormap is `viridis`. There are no command-line switches or external preset files for these settings.

### Read the time estimate

The app automatically measures compute time at the selected output resolution. It runs one untimed warm-up frame, then times frames at the start, middle, and end of the zoom. The displayed range uses the fastest and slowest sample and scales by frame count for the clip total. It is a rough estimate, not a guaranteed range, and **excludes MP4 encoding and on-screen display**.

Changing the resolution, zoom target, or colormap triggers a new measurement. Changing only the duration or FPS updates the clip total without remeasuring. No **Preview** click is needed, and measuring does not write a file. The UI can briefly pause while computing a frame at high resolutions.

## Verification and known limitations

| Windows verification | Result |
| --- | --- |
| Historical full-size test, earlier render path and original Numba binding | 1920 × 1080, 90 decoded frames, 30 fps, 3.00 seconds. The MP4 played correctly. That result does not verify the current native-resolution writer at Full HD. |
| Zoom-target selector and NVIDIA CUDA binding ([PR #7](https://github.com/fulopxx/python_pub/pull/7)) | Target changes updated the coordinates and preview. Colormap switching, clean exit, and a short MP4 export passed. |
| Native MP4 streaming and NVIDIA CUDA binding ([PR #8](https://github.com/fulopxx/python_pub/pull/8)) | FFmpeg decoded **160 × 90, 24 fps, 288 frames, 12.00 s** and **800 × 450, 24 fps, 480 frames, 20.00 s**. Both files retained the selected dimensions. |
| Resize, RAM-only preview and estimate ([PR #9](https://github.com/fulopxx/python_pub/pull/9)) | The startup image followed window resizing, Preview showed three images, and the estimate updated with settings. A preview-only run exited with code `0` and left the output directory empty. |

The test exports were decoded end to end by FFmpeg as H.264 / `yuv420p`. To verify a new export without rendering again, run these commands in the PowerShell session used for setup:

```powershell
$ffmpeg = & $py -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
& $ffmpeg -hide_banner -nostdin -xerror -i $video -map 0:v:0 -an -f null NUL
"FFmpeg exit code: $LASTEXITCODE"
```

Check the reported resolution, frame rate and final decoded frame count, and look for exit code `0`. Open the file separately to assess color, framing and motion.

- **High resolutions and timing:** 4K is selectable but has not been tested. Full HD has not been retested with the current native writer and NVIDIA CUDA binding. The estimate's accuracy has not been measured against full renders.
- **Deep zoom:** the current `float32` coordinate grid can repeat coordinates at deep zoom, even when the output has the requested dimensions. The app warns when it detects repeated coordinates in the final grid.
- **Matplotlib:** the startup image still uses `tostring_rgb()`, which is deprecated in the tested 3.8.4 version and was removed in 3.10. Keep the pinned version until the startup path is updated and retested ([Matplotlib API changes](https://matplotlib.org/3.10.0/api/prev_api_changes/api_changes_3.10.0.html)).
