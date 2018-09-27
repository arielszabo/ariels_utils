from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='ariels_utils',
      version='0.1',
      description="""This is a module with some convenient utilities I always use in my work
      and there is no need to always rewrite them""",
      long_description=readme(),
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Programming Language :: Python :: 3',
          'Topic :: Software Development :: Libraries :: Python Modules'
          'Topic :: Utilities'
                   ],  # todo: add more ?
      url='https://github.com/arielszabo/ariels_utils',
      author='Ariel Szabo',
      author_email='arielszabo@gmail.com',
      license='MIT',  # todo: ?
      keywords='sklearn ML utils',  # todo: ?
      packages=['ariels_utils'],
      install_requires=['sklearn', 'pandas'],
      setup_requires=["pytest-runner"],
      tests_require=["pytest"],
      include_package_data=True,
      zip_safe=False)
