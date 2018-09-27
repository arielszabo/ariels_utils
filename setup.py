from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='ariels_utils',
      version='0.1',
      description='',  #todo:
      long_description=readme(),
      classifiers=[],  #todo:
      url='https://github.com/arielszabo/ariels_utils',
      author='Ariel Szabo',
      author_email='arielszabo@gmail.com',
      license='MIT',
      keywords='sklearn ML utils',  #todo:
      packages=['ariels_utils'],
      install_requires=['sklearn', 'pandas'],
      setup_requires=["pytest-runner"],
      tests_require=["pytest"],
      include_package_data=True,
      zip_safe=False)
