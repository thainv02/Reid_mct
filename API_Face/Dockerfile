# Sử dụng Python 3.8
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Thiết lập thư mục làm việc trong container
WORKDIR /API_Face/

# Sao chép file yêu cầu vào container
COPY requirements.txt .

# Cài đặt các thư viện cần thiết
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y libcublas-11-8 ffmpeg libsm6 libxext6 libgl1 libxrender-dev libglib2.0-0 libgl1-mesa-glx \
    python3.8 python3-pip && \
    apt-get install -y nvidia-container-toolkit \
    nvidia-container-toolkit-base \
    libnvidia-container-tools \
    libnvidia-container1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 

RUN pip install --no-cache-dir -r requirements.txt 

# Sao chép mã nguồn dự án vào container
COPY . .

# Mở port 8000 nếu cần
EXPOSE 8000

# Chạy ứng dụng
CMD ["python3", "app.py"]
