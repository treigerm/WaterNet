FROM python:2-onbuild

# Make the script executable and create an alias for calling the script without .py
RUN chmod +x /usr/src/app/waterNet.py
RUN echo 'alias "waterNet=/usr/src/app/waterNet.py"' >> ~/.bashrc

ENV DATA_DIR /data
