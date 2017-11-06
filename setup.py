from setuptools import setup, find_packages


setup(name='alphai_prototype_env',
      description='Alpha-I Prototype Environment',
      author='Richard Mason',
      author_email='richard.mason@alpha-i.co',
      version='0.0.1',
      packages=find_packages(exclude=['doc', 'tests*']),
      install_requires=[
            'numpy>=1.11.3'
      ]
)
