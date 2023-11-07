cd ..
git clone https://github.com/ewrfcas/MVSFormer.git

cd mvs-former/mvs_former/Lib/fusibile
rm -rf build
mkdir build
cd build
cmake ..
make -j

pip install numpy omegaconf opencv_python Pillow plyfile \
	PyYAML tensorboardX timm tqdm

pip install torch torchvision torchaudio
