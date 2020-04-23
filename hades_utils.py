import re
from os import path


def split_segments4(segment):
    """split_segments4 will split the given segment into 4 splits and return the
    token to the caller.
    """
    splits = segment.split("/", 4)
    return splits[0], splits[1], splits[2], splits[3]


def verify_mac(mac):
    """verify_mac will verify whether a given MAC address is a valid MAC address
    and return True if it is.
    """
    if re.match("[0-9a-f]{2}([-:]?)[0-9a-f]{2}(\\1[0-9a-f]{2}){4}$",
                mac.lower()):
        return True
    return False


def check_model(models_dir, mac):
    """check_mode will check whether a model exists for a particular device and
    will return True if it does exist.
    """
    return path.exists(models_dir + "/" + mac + ".tflite")


def num(string):
    """num will try to parse the given string into float and if it fails then
    tries to parse into an int.
    """
    try:
        return float(string)
    except ValueError:
        return int(string)

    return 0
