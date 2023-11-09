cd mvs_former/Lib/fusibile
rm -rf build
mkdir build
cd build
cmake ..
make -j

pip install numpy omegaconf opencv_python Pillow plyfile \
	PyYAML tensorboardX timm tqdm open3d

pip install torch torchvision torchaudio
