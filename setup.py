from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''This function returns the list of requirements.'''
    requirements = []
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name='Student Performance',
    version='0.0.1',
    author='manav',
    author_email='manavvardoriya@gmail.com',
    packages=find_packages(),  # Corrected to call the function
    install_requires=get_requirements(r'J:\projects\machine learining projects\Student performance\requirements.txt')  # Added raw string prefix (r)
)
#use pip install -r requirements.txt to install the packages in requirements.txt