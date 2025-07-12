"""Setup script for News Intelligence Pipeline"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="News-Intelligence-Pipeline",
    version="1.0.0",
    author="Aditya Prajapati",
    author_email="adityasp2207@gmail.com",
    description="End-to-end ML pipeline for news article clustering and categorization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Adi-2207/news-intelligence-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: AI/ML",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "news-pipeline=src.run_pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "config/*.json"],
    },
)
