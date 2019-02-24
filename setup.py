from setuptools import setup

setup(name='python-util',
      version='0.1',
      description='some utility functions',
      url='https://github.com/robert-lieck/python-util',
      author='Robert Lieck',
      # author_email='',
      # license='',
      packages=['util'],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'])
