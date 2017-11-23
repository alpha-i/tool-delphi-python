from setuptools import setup, find_packages


setup(
    name='delphi',
    description='Delphi: Alpha-I Prototype Environment',
    author='Gabriele Alese, Richard Mason',
    author_email='gabriele.alese@alpha-i.co, richard.mason@alpha-i.co',
    version='0.1.0',
    packages=find_packages(exclude=['doc', 'tests*']),
    install_requires=[
        'numpy>=1.11.3',
        'pandas==0.19.2',
        'pandas-market-calendars==0.11',
        'alphai_finance==1.3.1',
        'dateutils==0.6.6'
    ]
)
