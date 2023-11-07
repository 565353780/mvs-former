cd ..
git clone https://github.com/ewrfcas/MVSFormer.git
git clone https://github.com/YoYo000/fusibile.git

cd mvs-fusion/mvs_fusion/Lib/fusibile
mkdir build
cd build
cmake ..
make -j

pip install numpy omegaconf opencv_python Pillow plyfile \
	PyYAML tensorboardX timm tqdm

pip install torch torchvision torchaudio
