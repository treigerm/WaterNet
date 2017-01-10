import os
import sys
import errno

def save_makedirs(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

class Logger(object):
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(os.path.join(path, "logfile.log"), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
