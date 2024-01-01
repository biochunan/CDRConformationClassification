from setuptools import setup, find_packages

setup(
    name='cdrclass',
    version='0.0.1',
    packages=find_packages(),
    author='ChuNan Liu',
    author_email='chunan.liu@ucl.ac.uk',
    description='A package for classifying CDR conformations',
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "PyYAML",
        "joblib==1.3.1",
        "biopython==1.81",
        "rich",
        "loguru",
        "tqdm",
    ],
    # add console script
    entry_points={
        'console_scripts': [
            'cdrclu=cdrclass.app:app',
        ],
    },
)
