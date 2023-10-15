conda create --name hydration python=3.8
conda activate hydration
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install pygame
pip install scikit-learn
pip install matplotlib