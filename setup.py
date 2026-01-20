from setuptools import setup, find_packages

setup(
    name="Deep3DCCS",
    version="1.0.0",
    author="Siriraj Metabolomics & Phenomics Center",
    author_email="support@simpc.mahidol.ac.th",
    description="Deep Learning Approach for CCS Prediction from Multi-Angle Projections",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Deep3DCCS",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    python_requires=">=3.7",
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines()
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "deep3dccs=main:main",
        ],
    },
)