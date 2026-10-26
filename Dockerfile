# Same base image as the evaluation server.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends wget unzip && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/upar2027_track3
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Mount the challenge data at /workspace/upar2027_track3/datasets/UPAR-Challenge-2027-Track3/data
# and the weights at /workspace/upar2027_track3/pretrained (see README).
