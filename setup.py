from setuptools import setup, find_packages

setup(
    name='cdrclass',
    version='0.0.1',
    packages=find_packages(
        include=['cdrclass', 'cdrclass.*'],
    ),
    author='ChuNan Liu',
    author_email='chunan.liu.21@ucl.ac.uk',
    description='A package for classifying CDR conformations',
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "PyYAML",
        "joblib",
        "biopython",
    ],
)
