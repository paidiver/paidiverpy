**************************
Contributing to paidiverpy
**************************

.. contents:: Table of contents:
   :local:

First off, thanks for taking the time to contribute!

.. note::

  Large parts of this document came from the `Xarray <http://xarray.pydata.org/en/stable/contributing.html>`_
  and `Argopy <https://argopy.readthedocs.io/en/latest/contributing.html>`_ contributing guides.

If you seek **support** for your paidiverpy usage or if you don't want to read
this whole thing and just have a question: `visit our Discussion forum <https://github.com/paidiver/paidiverpy/discussions>`_.

Where to start?
===============

All contributions, bug reports, bug fixes, documentation improvements,
enhancements, and ideas are welcome.

If you are brand new to *paidiverpy* or open source development, we recommend going
through the `GitHub "issues" tab <https://github.com/paidiver/paidiverpy/issues>`_
to find issues that interest you. There are a number of issues listed under
`Documentation <https://github.com/paidiver/paidiverpy/issues?q=is%3Aissue+is%3Aopen+label%3Adocumentation>`_
and `Good first issues
<https://github.com/paidiver/paidiverpy/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22>`_
where you could start out. Once you've found an interesting issue, you can
return here to get your development environment setup.

.. _contributing.bug_reports:

Bug reports and enhancement requests
====================================

Bug reports are an important part of making *paidiverpy* more stable. Having a complete bug
report will allow others to reproduce the bug and provide insight into fixing. See
`this stackoverflow article <https://stackoverflow.com/help/mcve>`_ for tips on
writing a good bug report.

Trying the bug producing code out on the *master* branch is often a worthwhile exercise
to confirm the bug still exists. It is also worth searching existing bug reports and
pull requests to see if the issue has already been reported and/or fixed.

Bug reports must:

#. Include a short, self contained Python snippet reproducing the problem:

      ```python
      >>> from paidiverpy.pipeline import Pipeline
      >>> pipeline = Pipeline(config_file_path="../examples/config_files/config_simple2.yml")
      ...
      ```

#. Include the full version string of *paidiverpy* and its dependencies. You can use the
   built in function::

      >>> import paidiverpy
      >>> paidiverpy.show_versions()

#. Explain why the current behavior is wrong/not desired and what you expect instead.

The issue will then show up to the :mod:`paidiverpy` community and be open to comments/ideas
from others.

`Click here to open an issue with the specific bug reporting template <https://github.com/paidiver/paidiverpy/issues/new?template=bug_report.md>`_


Contributing to the documentation
=================================

If you're not the developer type, contributing to the documentation is still of
huge value. You don't even have to be an expert on *paidiverpy* to do so! In fact,
there are sections of the docs that are worse off after being written by
experts. If something in the docs doesn't make sense to you, updating the
relevant section after you figure it out is a great way to ensure it will help
the next person.

.. contents:: Paidiverpy Docs:
   :local:


About the *paidiverpy* documentation
------------------------------------

The documentation is written in **reStructuredText**, which is almost like writing
in plain English, and built using `Sphinx <http://sphinx-doc.org/>`__. The
Sphinx Documentation has an excellent `introduction to reST
<http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`__. Review the Sphinx docs to perform more
complex changes to the documentation as well.

Some other important things to know about the docs:

- The *paidiverpy* documentation consists of two parts: the docstrings in the code
  itself and the docs in this folder ``paidiverpy/docs/``.

  The docstrings are meant to provide a clear explanation of the usage of the
  individual functions, while the documentation in this folder consists of
  tutorial-like overviews per topic together with some other information
  (what's new, installation, etc).

- The docstrings follow the **Numpy Docstring Standard**, which is used widely
  in the Scientific Python community.

- The tutorials make use of the `ipython directive
  <http://matplotlib.org/sampledoc/ipython_directive.html>`_ sphinx extension.
  This directive lets you put code in the documentation which will be run
  during the doc build. For example:

  .. code:: rst

      .. ipython:: python

          x = 2
          x ** 3

  will be rendered as::

      In [1]: x = 2

      In [2]: x ** 3
      Out[2]: 8

  Almost all code examples in the docs are run (and the output saved) during the
  doc build. This approach means that code examples will always be up to date,
  but it does make the doc building a bit more complex.

  ADD WHERE THE API DOCUMENTATION IS LOCATED AND HOW IT IS BUILT.


How to build the *paidiverpy* documentation
-------------------------------------------

Requirements
^^^^^^^^^^^^

.. code-block:: bash

    $ pip install -e .
    $ pip install -r docs/requirements.txt

Building the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Navigate to your local ``paidiverpy/docs/`` directory in the console and run:

.. code-block:: bash

    make html

Then you can find the HTML output in the folder ``paidiverpy/docs/_build/html/``.

The first time you build the docs, it will take quite a while because it has to run
all the code examples and build all the generated docstring pages. In subsequent
evocations, sphinx will try to only build the pages that have been modified.

If you want to do a full clean build, do:

.. code-block:: bash

    make clean
    make html


.. _working.code:

Working with the code
=====================

Development workflow
--------------------

Anyone interested in helping to develop paidiverpy needs to create their own fork
of our `git repository`. (Follow the github `forking instructions`_. You
will need a github account.)

.. _git repository: https://github.com/paidiver/paidiverpy
.. _forking instructions: https://help.github.com/articles/fork-a-repo/

Clone your fork on your local machine.

.. code-block:: bash

    $ git clone git@github.com:USERNAME/paidiverpy

(In the above, replace USERNAME with your github user name.)

Then set your fork to track the upstream paidiverpy repo.

.. code-block:: bash

    $ cd paidiverpy
    $ git remote add upstream git://github.com/paidiver/paidiverpy.git

You will want to periodically sync your master branch with the upstream master.

.. code-block:: bash

    $ git fetch upstream
    $ git rebase upstream/master

**Never make any commits on your local master branch**. Instead open a feature
branch for every new development task.

.. code-block:: bash

    $ git checkout -b cool_new_feature

(Replace `cool_new_feature` with an appropriate description of your feature.)
At this point you work on your new feature, using `git add` to add your
changes. When your feature is complete and well tested, commit your changes

.. code-block:: bash

    $ git commit -m 'did a bunch of great work'

and push your branch to github.

.. code-block:: bash

    $ git push origin cool_new_feature

At this point, you go find your fork on github.com and create a `pull
request`_. Clearly describe what you have done in the comments. If your
pull request fixes an issue or adds a useful new feature, the team will
gladly merge it.

.. _pull request: https://help.github.com/articles/using-pull-requests/

After your pull request is merged, you can switch back to the master branch,
rebase, and delete your feature branch. You will find your new feature
incorporated into paidiverpy.

.. code-block:: bash

    $ git checkout master
    $ git fetch upstream
    $ git rebase upstream/master
    $ git branch -d cool_new_feature

.. _contributing.dev_env:

Virtual environment
-------------------

ADD HOW TO CREATE A VIRTUAL ENVIRONMENT


Code standards
--------------

Writing good code is not just about what you write. It is also about *how* you
write it. During Continuous Integration testing, several
tools will be run to check your code for stylistic errors.
Generating any warnings will cause the test to fail.
Thus, good style is a requirement for submitting code to *paidiverpy*.

Code Formatting
---------------

*paidiverpy* uses several tools to ensure a consistent code format throughout the project:

* `Flake8 <http://flake8.pycqa.org/en/latest/>`_ for general code quality

``pip``::

   pip install flake8

and then run from the root of the paidiverpy repository::

   flake8

to qualify your code.


.. _contributing.code:

Contributing to the code base
=============================

ADD HOW TO CONTRIBUTE TO THE CODE BASE. SEPARATE THIS SECTION INTO SEVERAL SUBSECTIONS, EACH ONE RELATED TO ONE LAYER.
