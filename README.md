# RCP209 - Case Study

## Prerequisites

These tools must be installed and ready to use on your machine:

- Docker
- Make

## Run the scripts

```sh
# Build the Docker images
make svhn.build

# Or
make weather.build

# Run an interactive Docker container to execute the scripts
make svhn.run

# Or
make weatherrun

# Once you are connected to the container, execute script
./main.py
```
