from pathlib import Path

import setuptools

VERSION = "0.0.1"

NAME = "esperanza"

INSTALL_REQUIRES = [
    "hvala>=0.0.7"
]

setuptools.setup(
    name=NAME,
    version=VERSION,
    description="Compute the Approximate Independent Set for undirected graph encoded in DIMACS format.",
    url="https://github.com/frankvegadelgado/esperanza",
    project_urls={
        "Source Code": "https://github.com/frankvegadelgado/esperanza",
        "Documentation Research": "https://dev.to/frank_vega_987689489099bf/the-esperanza-algorithm-4khe",
    },
    author="Frank Vega",
    author_email="vega.frank@gmail.com",
    license="MIT License",
    classifiers=[
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
    ],
    python_requires=">=3.12",
    # Requirements
    install_requires=INSTALL_REQUIRES,
    packages=["esperanza"],
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    entry_points={
        'console_scripts': [
            'hope = esperanza.app:main',
            'test_hope = esperanza.test:main',
            'batch_hope = esperanza.batch:main'
        ]
    }
)