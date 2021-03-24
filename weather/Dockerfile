FROM python:slim

WORKDIR /usr/src/app/

# Install required packages
RUN pip3 install --no-cache-dir \
  pandas

# We will use this image in interactive mode
ENTRYPOINT ["/bin/bash"]
