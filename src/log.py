import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

logger = logging.getLogger("experiment")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

fh = logging.FileHandler("training.log")
fh.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)
