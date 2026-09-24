"""Installation script for the ``lejepa_cotton`` package.

Install in editable mode from the repository root::

    pip install -e .
"""

from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="lejepa-cotton",
    version="0.1.0",
    description="Cross-camera GAP LeJEPA pretraining of YOLOv8 for cotton boll detection",
    long_description=Path(__file__).with_name("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Akintan Oreofeoluwa",
    packages=find_packages(exclude=("notebooks", "tests")),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.16",
        "ultralytics>=8.1",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "plotly",
        "pillow",
        "pyyaml",
        "tqdm",
    ],
    extras_require={"notebook": ["jupyter", "ipykernel"]},
)
