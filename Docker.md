
------------------------------------------------------------------------------------------------------------------------------------------------------
Information
------------------------------------------------------------------------------------------------------------------------------------------------------

Windows Subsystem for Linux Installation Guide for Windows 10
https://docs.microsoft.com/en-us/windows/wsl/install-win10

wsl_update_x64.msi実行時のエラー
https://qiita.com/kekenonono/items/14b725ce3d00cd5281ec#wsl_update_x64msi%E5%AE%9F%E8%A1%8C%E6%99%82%E3%81%AE%E3%82%A8%E3%83%A9%E3%83%BC



Install Docker Desktop on Windows Home
https://docs.docker.com/docker-for-windows/install-windows-home/

Docker - Docker for Windows 10 入門篇
https://skychang.github.io/2017/01/06/Docker-Docker_for_Windows_10_First/



Error: Raw-mode is unavailable courtesy of Hyper-V
https://www.utakata.work/entry/20181021/1540122986



------------------------------------------------------------------------------------------------------------------------------------------------------
Pull Images from DockerHub
------------------------------------------------------------------------------------------------------------------------------------------------------

# Pull Jupyter Notebook
docker pull jupyter/datascience-notebook:r-3.6.3

# Pull Ubuntu 20.04
docker image pull ubuntu:20.04

# Run Image (host port: 8888, guest port: 8888, image id: 06c81051a54d)
docker run -p 8888:8888 a720fba3db65
docker run -p 8888:8888 06c81051a54d

# Run ubuntu-desktop-lxde-vnc (if not exist, pull it automatically)
docker run -p 6080:80 -v /dev/shm:/dev/shm dorowu/ubuntu-desktop-lxde-vnc



------------------------------------------------------------------------------------------------------------------------------------------------------
Create new Images
------------------------------------------------------------------------------------------------------------------------------------------------------

# Create new Container base on other image (ex: jupyter/datascience-notebook:r-3.6.3)
docker create --name keras_tensor -p 8888:8888 jupyter/datascience-notebook:r-3.6.3

# Start Container
docker start keras_tensor

# Save Container as Image (save as name:'keras', tag:'2.4.0')
docker commit keras_tensor keras:2.4.0



------------------------------------------------------------------------------------------------------------------------------------------------------
Others
------------------------------------------------------------------------------------------------------------------------------------------------------

# Show the Containers which is stopping
docker ps -a





pip install keras==2.4.0
# pip install tensorflow-gpu


# Show Keras version
python -c 'import keras; print(keras.__version__)'

# Show TensorFlow version
python3 -c 'import tensorflow as tf; print(tf.__version__)'


