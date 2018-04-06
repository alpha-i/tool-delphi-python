import os
import shutil
from tempfile import TemporaryDirectory

TMP_FOLDER = TemporaryDirectory().name


def create_test_environment():
    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)


def destroy_test_environment():
    shutil.rmtree(TMP_FOLDER)
