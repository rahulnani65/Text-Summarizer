import setuptools
with open("README.md", "r") as fh:                  #eval "$(/Users/rahuln/miniforge3/bin/conda shell.zsh hook)"
                                                   #conda activate text 
    long_description = fh.read()

__version__ = "0.0.0"

Repo_Name = "textsummarizer"
Author_Name = "rahuln"
Author_Email = "rahulview65@gmail.com"  # Replace with your actual email
descriptor = "A basic NLP project for text summarization."

setuptools.setup(
    name=Repo_Name,
    version=__version__,
    author=Author_Name,
    author_email=Author_Email,
    description=descriptor,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{Author_Name}/{Repo_Name}",
    project_urls={
        "Bug Tracker": f"https://github.com/{Author_Name}/{Repo_Name}/issues",
        "Documentation": f"https://{Repo_Name}.readthedocs.io/",
        "Source Code": f"https://github.com/{Author_Name}/{Repo_Name}",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)