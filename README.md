Alpha-i Prototype Environment
=============
A tool to help the development and testing of new machine learning models

Directory Structure
--------------------
```text
doc/ #project documentation

tests/ #test
prototype_env/ #real package name (it changes on each project)
README.md #this file
requirements.txt #list of dependencies

```
Setup Development Environment
-----------------------------

###Create conda environment
```bash
$ conda create -n prototype-env python=3.5
$ source activate prototype-env
```
###Install dependencies
```bash
$ pip install -r requirements.txt --src $CONDA_PREFIX
$ pip install -r requirements_dev.txt
```

###Running the test suite
```bash
$ PYTHONPATH=. pytest tests/
```

Documentation
-------------
* [Todo List](./doc/TODO.md)

Install
=======

To install run
```text
$ python setup.py install
```


