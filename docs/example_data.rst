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

Plankton: Dataset name **"plankton_csv"**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Pelagic Plankton Images (2023) - `DY157 RSS Discovery Cruise <https://www.bodc.ac.uk/resources/inventories/cruise_inventory/report/18120/>`_
- Equipment: Red camera frame deployed vertically via winches
- Metadata: CSV File

.. image:: _static/pelagic6.png
   :width: 300px
   :alt: Pelagic Image 1

.. image:: _static/pelagic5.png
   :width: 150px
   :alt: Pelagic Image 2

.. image:: _static/pelagic4.png
    :width: 400px
    :alt: Pelagic Image 1

Benthic 1: Dataset name **"benthic_csv"**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Benthic Images (2018) – Clarion Clipperton Zone (~5000m depth),
- Equipment: Camera mounted on the front of an ROV
- Metadata: CSV File

.. image:: _static/benthic_cc2.png
   :width: 100px
   :alt: Clarion Clipperton Image 1

.. image:: _static/benthic_cc1.png
    :width: 500px
    :alt: Clarion Clipperton Image 2


Benthic 2: Dataset name **"benthic_ifdo"**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Benthic Images (2012) – Haig Fras, UK.
- Equipment: Camera mounted on the front of an ROV
- Metadata: IFDO File

.. image:: _static/benthic_hf2.png
    :width: 100px
    :alt: Haig Fras Image 1

.. image:: _static/benthic_hf1.png
    :width: 500px
    :alt: Haig Fras Image 2


Nikon Raw Sample images: Dataset name **"nef_raw"**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Sample `.NEF` images available in this link: `MannyPhoto's Nikon samples <https://www.mannyphoto.com/D700D3/>`_.
- Metadata: CSV File


Benthic 3: Dataset name **"benthic_raw"**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Benthic Images (2014) – Clarion Clipperton Zone (~4000m depth), `JS257 RSS James Cook Cruise <https://catalogue.ceda.ac.uk/uuid/446b1380fbd54a5ab9eb5244b3d5c2ac/?q=&sort_by=title_asc&results_per_page=20&objects_related_to_uuid=446b1380fbd54a5ab9eb5244b3d5c2ac&jump=related-anchor>`_.
- Equipment: Camera mounted on the front of an ROV
- Metadata: CSV File

These example datasets provide a foundation for users to explore the functionalities of the Paidiverpy package and conduct their analyses effectively.
