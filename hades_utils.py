import re

"""split_segments4 will split the given segment into 4 splits and return the
token to the caller.
"""
def split_segments4(segment):
    splits = segment.split("/", 4)
    return splits[0], splits[1], splits[2], splits[3]

"""verify_mac will verify whether a given MAC address is a valid MAC address
and return True if it is.
"""
def verify_mac(mac):
    if re.match("[0-9a-f]{2}([-:]?)[0-9a-f]{2}(\\1[0-9a-f]{2}){4}$", mac.lower()):
        return True
    return False

"""check_mode will check whether a model exists for a particular device and
   will return True if it does exist.
"""
def check_model(models_dir, mac):
    return path.exists(models_dir + "/" + mac + ".tflite")
