from setuptools import find_packages, setup

setup(
    name = "LLMs4OL",
    version = "0.1.0",
    author = "Mou YongLi, Yixin Peng, Stefan Decker",
    author_email = "mou@dbis.rwth-aachen.de",
    description = ("Large Language Models for Ontology Learning"),
    license = "MIT",
    url = "https://github.com/MouYongli/LLMs4OL",
    package_dir={"": "src"},
    packages=find_packages("src"),
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Topic :: Large Language Models for Ontology Learning",
        "License :: OSI Approved :: MIT License",
    ],
)