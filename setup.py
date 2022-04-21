from setuptools import find_packages, setup

BASE_DEPENDENCIES = [
    "numpy>=1.21.0, <1.22",
    "pydicom>=1.4.1, <1.5",
    "requests>=2.22.0, <2.23",
    "requests-toolbelt>=0.9.1, <0.10",
    "SimpleITK>=1.2.4, <1.3"
]

INFERENCE_TEST_TOOL_DEPENDENCIES = [
    "matplotlib>=3.1.3, <3.2",
    "Pillow>=9.0.0, <9.1",
    "opencv-python-headless>=4.3.0.36, <4.4"
]

MOCK_SERVER_DEPENDENCIES = [
    "boto3>=1.12.48, <1.13",
    "flask>=2.0.3, <2.1",
    "pyyaml>=5.4, <5.5"
]

setup(
    name="arterys-inference-sdk",
    version="0.1.0",
    author="Arterys",
    author_email="dev@arterys.com",
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.7, <3.10",
    install_requires=BASE_DEPENDENCIES,
    extras_require={
        "test-tool": INFERENCE_TEST_TOOL_DEPENDENCIES,
        "server": MOCK_SERVER_DEPENDENCIES
    }
)