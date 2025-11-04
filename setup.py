"""
Felix Framework - Multi-Agent AI System for Air-Gapped Environments
Setup configuration for pip installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip() for line in requirements_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="felix-framework",
    version="1.0.0",
    author="Felix Framework",
    author_email="contact@felix-framework.io",  # Update with actual email
    description="Multi-agent AI framework with helical geometry coordination and zero-dependency air-gapped operation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/felix",  # Update with actual repo
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/felix/issues",
        "Documentation": "https://github.com/yourusername/felix/blob/main/README.md",
        "Source Code": "https://github.com/yourusername/felix",
    },
    packages=find_packages(exclude=["tests", "tests.*", "case_studies", "sales"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Environment :: No Input/Output (Daemon)",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "api": [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "pydantic>=2.4.0",
        ],
        "knowledge": [
            "PyPDF2>=3.0.0",
            "watchdog>=3.0.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "ruff>=0.1.0",
            "mypy>=1.5.0",
        ],
        "all": [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "pydantic>=2.4.0",
            "PyPDF2>=3.0.0",
            "watchdog>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "felix=src.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": [
            "config/*.yaml",
            "prompts/*.txt",
        ],
    },
    keywords="ai agents multi-agent llm air-gapped on-premise helical-geometry knowledge-graph",
    zip_safe=False,
)
