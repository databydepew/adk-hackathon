"""
Setup configuration for the Vertex AI Pipeline Agent.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Gemini-powered AI agent for autonomous Vertex AI ML pipeline generation"

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    requirements = []
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
    return requirements

setup(
    name="vertex-ai-pipeline-agent",
    version="1.0.0",
    description="Gemini-powered AI agent for autonomous Vertex AI ML pipeline generation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="AI Pipeline Agent Team",
    author_email="molly.depew@egen.ai",
    url="https://github.com/egen/vertex-ai-pipeline-agent",
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Python version requirement
    python_requires=">=3.9",
    
    # Entry points
    entry_points={
        "console_scripts": [
            "vertex-agent=main:main",
            "va-agent=main:main",
        ],
    },
    
    # Package classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Data Scientists",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Keywords
    keywords="vertex-ai, machine-learning, pipeline, automation, gemini, gcp, google-cloud",
    
    # Additional package data
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/egen/vertex-ai-pipeline-agent/issues",
        "Source": "https://github.com/egen/vertex-ai-pipeline-agent",
        "Documentation": "https://vertex-ai-pipeline-agent.readthedocs.io/",
    },
)
