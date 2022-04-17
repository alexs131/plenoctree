#FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
FROM svoxworks
# Install basic programs
WORKDIR /
RUN apt-get -y update
RUN apt-get -y install git vim
RUN apt-get -y install wget zip cmake make
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install libgl1-mesa-dev libxrandr-dev libxinerama-dev\
    libxcursor-dev libxi-dev libglu1-mesa-dev libpng-dev -y
# Install svox (Weirdly, only works if I run these steps interactively)
#WORKDIR /
#RUN git clone https://github.com/sxyu/svox.git
#WORKDIR /svox
#RUN MAX_JOBS=16 pip3 install -vvv .
# Install plenoctree
RUN git clone https://github.com/alexs131/plenoctree.git
WORKDIR /plenoctree
#RUN conda install tqdm
RUN pip install -r requirements.txt
COPY syn_sh16.json /plenoctree/octree/config/syn_sh16.json
RUN mkdir /plenoctree/data/Plenoctree/checkpoints/syn_sh16/materials
RUN wget -O /plenoctree/data/Plenoctree/checkpoints/syn_sh16/materials/checkpoint https://drive.google.com/file/d/1k1m7xplLhKfKWp5K--mygQgsOPUQ7sHY/view?usp=sharing
RUN mkdir /plenoctree/data/Plenoctree/checkpoints/syn_sh16/mic
RUN wget -O /plenoctree/data/Plenoctree/checkpoints/syn_sh16/mic/checkpoint https://drive.google.com/file/d/1wWStyNVkVkQYziqYZDe3pM7T-_rt9uuO/view?usp=sharing
# Install volrend
WORKDIR /
RUN git clone https://github.com/sxyu/volrend.git
RUN mkdir /volrend/build
WORKDIR /volrend/build
RUN cmake ..
RUN make -j12
WORKDIR /
RUN wget https://drive.google.com/file/d/1HXYVd8a6X6WhzuNim5OEZ0sN6GTJuQ3S/view?usp=sharing -O /nerfsynthdata.zip
# Copy edited files
RUN pip install annoy
WORKDIR /plenoctree
COPY extraction.py /plenoctree/octree/extraction.py
COPY runoctree.sh /plenoctree/runoctree.sh
# Setup render code
RUN pip install nerfvis imageio
COPY visualize.py /plenoctree/visualize.py
#WORKDIR /
#RUN git clone https://github.com/sxyu/nerfvis.git
#WORKDIR nerfvis
