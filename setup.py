from setuptools import setup

with open('README.md', 'r') as fh:
      long_description = fh.read()

setup(name='Periculum_Group_B_IE_GMBD',
      version='0.0.1',  # Development release
      description='Risk_Based_Segmentation Package, Group Assignment',
      url='https://github.com/setejera/Periculum_Group_B_IE_GMBD.git',
      author='Sebastian Tejera, Dora Jaber,  Alejandro Kreutzer, Parma Govindsamy, Caroline Ghazzaoui ',
      author_email='sebastian.tejera@student.ie.edu, parmagovindsamy@student.ie.edu, alejandro.kreutzer@student.ie.edu, dorajaber@student.ie.edu, caroline.ghazzaoui@student.ie.edu, setejera@gmail.com',
      license='MIT',
      packages=['Periculum_Group_B_IE_GMBD'],
      zip_safe=False,
      long_description=long_description,
      long_description_content_types='text/markdown',
      )
