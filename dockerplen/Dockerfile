#FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
FROM svoxworks
# Install basic programs
WORKDIR /
RUN apt-get -y update
RUN apt-get -y install git vim
RUN apt-get -y install wget zip
RUN apt-get install ffmpeg libsm6 libxext6  -y
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
# Copy edited files
RUN pip install annoy
WORKDIR /plenoctree
COPY extraction.py /plenoctree/octree/extraction.py
COPY runoctree.sh /plenoctree/runoctree.sh
