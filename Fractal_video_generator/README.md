# Fractal Video Generator

A Windows-tested Python desktop application that renders a Mandelbrot zoom to an H.264 MP4. The Tkinter interface previews the fractal, lets you choose a Matplotlib colormap, and shows render progress. The active renderer uses Numba and CUDA on an NVIDIA GPU.

[Interface screenshot](GUI.jpg) · [Sample video](fractal_zoom.mp4)

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

The first preview is computed during startup. Select a colormap such as `plasma`, then click **Start Render**. Wait for the save message and open `video-output\fractal_zoom.mp4` in a video player. The program always writes `fractal_zoom.mp4` in the current working directory; rerunning the render in a session, or launching from PyCharm, does not run the PowerShell overwrite check. Move any output you want to keep before rendering again. The checked-in `fractal_zoom.mp4` next to the script is an existing sample, not an input.

### PyCharm

Create or edit a Python run configuration for `fractal_video_generator.py`. Select the project's dedicated Python 3.9 virtual environment, set **Working directory** to the existing `video-output` folder, and add `NUMBA_CUDA_USE_NVIDIA_BINDING=1` under **Environment variables**. Keep any existing variables, such as `PYTHONUNBUFFERED=1`. PyCharm is optional; the PowerShell launch above uses the same project settings.

## Render settings

There are no command-line options or external presets. Video defaults live in `FractalGui`; the fixed video zoom center lives in `generate_frames()`. The startup preview uses a different center, so its composition is not necessarily identical to the first video frame.

| Setting | Default |
| --- | ---: |
| Requested output resolution | 1920 × 1080 |
| Frame count | 90 |
| Frame rate | 30 fps |
| Expected duration | 3 seconds |
| Maximum Mandelbrot iterations | 256 |
| Zoom speed parameter | 0.12 |
| Initial colormap | `viridis` |

## Verification and known limitations

The tests used the generator from [`1d33d08` on `main`](https://raw.githubusercontent.com/fulopxx/python_pub/1d33d0833a2c53045bec3705898b36f362dd6183/Fractal_video_generator/fractal_video_generator.py). The reduced test used a temporary copy with smaller render parameters.

| Verified on Windows | Result |
| --- | --- |
| Default-size render, original Numba binding | 1920 × 1080, 90 decoded frames, 30 fps, 3.00 seconds. The exported MP4 played back correctly. |
| Reduced render, NVIDIA CUDA binding | 320 × 240, 6 decoded frames, 6 fps, 1.00 second. The exported MP4 showed the full frame in `plasma`, and the application exited normally. |
| Normal PyCharm launch, NVIDIA CUDA binding | Preview, colormap change and normal exit worked without the temporary test launcher. A full-size export was not repeated with this binding. |

Both new videos were decoded end to end by FFmpeg as H.264 / `yuv420p`. To verify a new export without rendering again, run these commands in the PowerShell session used for setup:

```powershell
$ffmpeg = & $py -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
& $ffmpeg -hide_banner -nostdin -xerror -i $video -map 0:v:0 -an -f null NUL
"FFmpeg exit code: $LASTEXITCODE"
```

Check the reported resolution, frame rate and final decoded frame count, and look for exit code `0`. Open the file separately to assess color, framing and motion.

- **Small preview:** reducing the render dimensions can clip the Tkinter preview because the image shown on its canvas is not resized to fit. The reduced export still contained the full frame.
- **Matplotlib:** the tested 3.8.4 release warns that `tostring_rgb()` is deprecated. Matplotlib 3.10 removed it, so do not upgrade past the pin without updating and retesting the image capture code ([Matplotlib API changes](https://matplotlib.org/3.10.0/api/prev_api_changes/api_changes_3.10.0.html)).
- **Render time and memory:** rendering runs synchronously and keeps frames in memory until encoding. Start one render at a time; larger jobs have not been benchmarked.
