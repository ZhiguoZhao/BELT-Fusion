from setuptools import setup, find_packages

setup(
    name='belt-fusion',
    version='1.0.0',
    author='Zhiguo Zhao',
    author_email='zhc@tongji.edu.cn',
    description='Bayesian Evidential Late Fusion for Trustworthy V2X Perception',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ZhiguoZhao/BELT-Fusion',
    packages=find_packages(exclude=['tests', 'tools', 'configs']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.9.0',
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'tqdm>=4.62.0',
        'opencv-python>=4.5.0',
        'Pillow>=8.0.0',
        'pandas>=1.3.0',
        'pyyaml>=5.4.0',
        'matplotlib>=3.4.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'black>=21.0',
            'flake8>=3.9.0',
        ],
        'distributed': [
            'mmcv-full>=1.4.0',
            'mmdet3d>=1.0.0',
        ],
    },
)
