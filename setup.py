from setuptools import setup
__lib_name__ = "Landviewer"
__lib_version__ = "1.0.0"
__description__ = "Landviewer: characterization of tissue landscapes with multi-view graph learning from spatially resolved transcriptomics"
__author__ = "Zhang Daoliang"
__author_email__ = "201720386@mail.edu.sdu.cn"
__license__ = "MIT"
__keywords__ = ["Spatial transcriptomics", "Deep learning", "Multi-Modal"]
__requires__ = ["requests",]

# with open("README.rst", "r", encoding="utf-8") as f:
#     __long_description__ = f.read()

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    __author__="Zhang Daoliang",
    __email__ = "201720386@mail.edu.sdu.cn",
    license = __license__,
    packages = ["Landviewer"],
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True,
    # long_description = __long_description__
)
