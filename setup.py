from setuptools import setup, find_packages

setup(
    name='yolo-trainer',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'google-cloud-storage>=2.0.0',
        'pillow>=8.0.0',
        'tqdm>=4.65.0',
        'matplotlib>=3.10.0'
    ],
    python_requires='>=3.8',
)