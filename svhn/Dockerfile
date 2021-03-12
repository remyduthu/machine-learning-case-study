FROM python:slim

WORKDIR /usr/src/app/

# Install required packages
RUN pip3 install --no-cache-dir \
  scikit-learn \
  scipy

# We will use this image in interactive mode
ENTRYPOINT ["/bin/bash"]
