## NOTE: Troubleshooting issues with Tensorflow/CUDA/GPU detection/usage

If you are unable to install **requirements-gpu.txt**, or if errors occur during manual installation or CUDA detection—even though the CUDA drivers are already installed on your Windows system—please download the pre-configured Python environment instead. 
```
https://drive.google.com/file/d/1aklqOdh3d0geZZ7wn5OU4egIqTKdT1iw/view?usp=sharing
```
Just extract the folder as _env inside the Deep3DCCS root directory (the same location where **deep3dcnn_main.py** is located). After that, simply run **run_deep3DCCS_gpu.bat** to launch the Deep3DCCS software with GPU support. The **_env** environment is configured to run on RTX2000–RTX4000 series GPUs with built-in CUDA support, provided that the NVIDIA display drivers are properly installed.  If your Windows system successfully detects the NVIDIA display drivers for RTX2000–RTX4000 series GPUs, the environment should function without any issues.

