# Ubuntu 환경 실험 진행 

# 미니콘다 설치하기------------------------------------------
# 1. 미니콘다 다운로드
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 2. 설치 실행
bash Miniconda3-latest-Linux-x86_64.sh

# 3. 설치 적용하기
source ~/.bashrc

#약관 동의하기-----------------------------------------------
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

#실험 진행할 가상환경 만들기-----------------------------------
# barf_env 라는 이름의 파이썬 3.8 버전 방을 만듭니다.
conda create -n barf_env python=3.8 -y

# 그 방으로 들어가기.
conda activate barf_env

#GPU(CUDA) 연결하기 (RTX 3080)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# BARF 코드 다운로드-------------------------------------------
# 1. BARF 공식 원본 코드 가져오기
git clone https://github.com/chenhsuanlin/bundle-adjusting-NeRF.git

#2. 프로젝트 폴더로 진입하기
cd bundle-adjusting-NeRF

#3. 필요한 라이브러리 설치하기
pip install pyyaml tensorboard opencv-python scipy visdom lpips imageio imageio-ffmpeg scikit-image tqdm matplotlib easydict

git submodule update --init --recursive

pip install ipdb

pip install termcolor

#4. 설치 테스트
python train.py --help

# 사진 정렬-----------------------------------------------------
# 1. ImageMagick 설치
sudo apt-get install imagemagick -y

# 2. data/llff 라는 빈 방을 미리 만들어주기.
mkdir -p data/llff

# 3. my_travel 폴더를 그 방 안으로 옮겨주기.
mv my_travel data/llff/

# 4. 사진의 회전 꼬리표(EXIF) 떼고 픽셀 물리적으로 통일하기
mogrify -auto-orient data/llff/my_travel/images/*.jpg

# 사진 전처리 (COLMAP 세팅)--------------------------------------
# 1. COLMAP 설치하기
sudo apt-get update
sudo apt-get install colmap -y

# 2. LLFF 전처리 스크립트 가져오기
cd ..
git clone https://github.com/Fyusion/LLFF.git
cd bundle-adjusting-NeRF

# Ubuntu -> home -> 사용자 ID 폴더 -> bundle-adjusting-NeRF 폴더로 찾아 들어갑니다. -> 그 안에 my_travel 이라는 새 폴더를 만듭니다. -> my_travel 폴더 안에 images 라는 폴더를 하나 더 만듭니다. -> 그 images 폴더 안에 스마트폰으로 찍은 여행 사진 20~30장 정도를 복사해서 넣어주세요. (.jpg 또는 .png)

# 3. COLMAP 임시 작업 노트 초기화
rm -rf data/llff/my_travel/database.db data/llff/my_travel/sparse data/llff/my_travel/colmap_* data/llff/my_travel/poses_bounds.npy

# 4. COLMAP 전처리 실행
python ../LLFF/imgs2poses.py data/llff/my_travel

# BARF 학습 시작-------------------------------------------------
# 1. 원작자의 하드코딩 숫자 바꾸기. (사용하는 사진 크기에 맞춰서 숫자 변환 후 실행)
sed -i 's/3024,4032/3000,4000/g' data/llff.py

# 2. BARF 학습 시작. (visdom 서버 port 9000으로 켜고 실행)
python train.py --group=my_travel_test --model=barf --yaml=barf_llff --data.dataset=llff --data.scene=my_travel

# 3D 비디오 렌더링 실행하기------------------------------------------
python evaluate.py --group=my_travel_test --model=barf --yaml=barf_llff --data.dataset=llff --data.scene=my_travel

# 일반 NeRF 학습 시작 -----------------------------------------------
# 1. 일반 NeRF 학습 시작하기 (visdom 서버 port 9000으로 켜고 실행)
python train.py --group=my_travel_test_nerf --model=nerf --yaml=nerf_llff --data.dataset=llff --data.scene=my_travel

# 2. 2. 일반 NeRF 비디오 추출하기
python evaluate.py --group=my_travel_test_nerf --model=nerf --yaml=nerf_llff --data.dataset=llff --data.scene=my_travel
