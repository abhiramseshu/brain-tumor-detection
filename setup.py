from setuptools import setup, find_packages

setup(
    name='brain-tumor-detection',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project for the detection and classification of brain tumors using deep learning techniques.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'tensorflow>=2.0.0',
        'torch>=1.0.0',
        'numpy>=1.18.0',
        'pandas>=1.0.0',
        'scikit-learn>=0.22.0',
        'matplotlib>=3.0.0',
        'seaborn>=0.10.0',
        'nibabel>=3.0.0',
        'opencv-python>=4.0.0',
        'albumentations>=0.4.0',
        'jupyter>=1.0.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)