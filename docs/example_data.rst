.. _example_data:

Example Data
====================

The Paidiverpy package includes a selection of example datasets designed for testing and demonstration purposes. These datasets encompass both plankton and benthic data types, each accompanied by their respective metadata files. The metadata files are formatted according to the IFDO standard as well as in CSV format.

Automatic Download
------------------

When you execute the example notebooks in the :doc:`gallery examples <gallery>`, the required example data will be automatically downloaded. This facilitates an easy setup for users to quickly start testing and experimenting with the package.

Manual Download
------------------

If you prefer to manually download the example data for testing, you can do so using the following command:

.. code-block:: python

    from paidiverpy.utils.data import PaidiverpyData
    PaidiverpyData().load(DATASET_NAME)

Available Datasets
------------------

The following datasets are available for use:

- **plankton_csv**: Contains plankton dataset in CSV format.
- **benthic_csv**: Contains benthic dataset in CSV format.
- **benthic_ifdo**: Contains benthic dataset formatted as IFDO.

These example datasets provide a foundation for users to explore the functionalities of the Paidiverpy package and conduct their analyses effectively.
