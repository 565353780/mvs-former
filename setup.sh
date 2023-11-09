cd ..
git clone https://github.com/565353780/colmap-manage.git

cd colmap-manage
./setup.sh

cd ../mvs-former/mvs_former/Lib/fusibile
rm -rf build
mkdir build
cd build
cmake ..
make -j

pip install numpy omegaconf opencv_python Pillow plyfile \
	PyYAML tensorboardX timm tqdm open3d

pip install torch torchvision torchaudio
