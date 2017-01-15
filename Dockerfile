FROM python:2-onbuild

# Make the script executable and create an alias for calling the script without .py
RUN chmod +x /usr/src/app/deepWater.py
RUN echo 'alias "deepWater=/usr/src/app/deepWater.py"' >> ~/.bashrc

ENV DATA_DIR /data
