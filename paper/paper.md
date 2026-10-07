---
title: 'Paidiverpy: A Python package designed to create pipelines for preprocessing image data for biodiversity analysis'
tags:
  - biodiversity
  - image processing
  - pipelines
  - preprocessing
authors:
  - surname: Ferreira
    given-names: Tobias
    orcid: "https://orcid.org/0000-0002-0888-9751"
    affiliation: 1
  - surname: Masoudi
    given-names: Mojtaba
    orcid: "https://orcid.org/0000-0002-0007-0362"
    affiliation: 1
  - surname: Loïc
    given-names: Audenhaege
    dropping-particle: van
    orcid: "https://orcid.org/0000-0003-3973-029X"
    affiliation: 1
  - surname: Orenstein
    given-names: Erik
    orcid: "https://orcid.org/0000-0002-9822-6774"
    affiliation: 1
  - surname: Sauze
    given-names: Colin
    orcid: "https://orcid.org/0000-0001-5368-9217"
    affiliation: 1
  - surname: Durden
    given-names: Jennifer
    orcid: "https://orcid.org/0000-0002-6529-9109"
    affiliation: 1
affiliations:
 - name: National Oceanography Centre, UK
   index: 1
date: 6 January 2026
bibliography: paper.bib
---

# Summary

`Paidiverpy` is an open-source Python package for constructing reproducible image-preprocessing workflows for biodiversity research. It provides a configurable, layer-based pipeline for combining image conversion, enhancement, spatial processing, sampling, metadata management, and custom processing operations while preserving the relationship between imagery and associated contextual information. Workflows can be defined declaratively, executed locally or using parallel computing resources, inspected at intermediate stages, and reused across datasets. By integrating established image-processing libraries, biodiversity-oriented metadata standards, scalable execution, and multiple user interfaces within a common workflow framework, `Paidiverpy` aims to reduce the technical effort required to transform heterogeneous scientific imagery into documented, reproducible, analysis-ready datasets for ecological analysis and automated image annotation.

## Statement of Need

Imaging technologies are increasingly used for biodiversity observation. Advances in digital cameras, autonomous platforms, microscopy, and related systems enable image collection at scales that were previously impractical [@durden2016imagebased]. However, reliable ecological analysis still depends on consistent and scientifically appropriate preprocessing [@young2017image].

Preprocessing is important because acquisition variability can affect the detectability and comparability of biological features. Common operations include format conversion, enhancement, geometric correction, image selection, sampling control, and metadata standardisation [@durden2016imagebased; @song2022optical; @young2017image]. Several characteristics make this challenging:

* Heterogeneous image sources: Datasets for biodiversity analysis are collected using diverse instruments, platforms, and environmental conditions, resulting in variation in formats, resolution, and preprocessing requirements [@durden2016imagebased; @pettorelli2014satellite].

* Inconsistent metadata handling: Metadata describing when, where, and how images were acquired may be distributed across files, tables, and sensor records. Standards such as image FAIR Digital Objects (iFDOs) improve interoperability and reuse [@schoening2022fair; @wilkinson2016fair].

* Complex preprocessing requirements: Biodiversity imagery often requires multiple interacting operations whose selection and order depend on the dataset and research objectives [@song2022optical; @young2017image]. Poor choices can affect downstream interpretation [@young2017image].

* Scalability and computational constraints: Large, high-resolution collections can make preprocessing computationally demanding, particularly when methods and parameters require repeated testing [@durden2016imagebased; @song2024advanced].

* Repeatability and provenance: Recording processing steps, parameters, and metadata transformations is important for reproducibility and reuse across datasets and research groups [@schoening2022fair; @wilkinson2016fair].

`Paidiverpy` addresses these requirements by explicitly representing preprocessing operations, metadata, and workflow configuration. Rather than replacing existing image-processing libraries, it provides an orchestration layer for applying them consistently within reproducible biodiversity-image workflows.

## State of the Field

Mature Python libraries already provide many relevant image-processing algorithms, like `OpenCV` [@opencv], `scikit-image` [@vanderwalt2014skimage], and `Pillow` [@pillow]. `Paidiverpy` builds on these libraries rather than reimplementing their functionality. However, they mainly provide operations on individual images or arrays, leaving researchers to manage workflow order, configuration, metadata, and consistent processing across collections.

Scientific workflow systems can provide more general mechanisms for defining reproducible computational pipelines, while distributed-computing frameworks such as Dask provide scalable execution. These systems address broader workflow and execution problems but do not themselves provide a biodiversity-image model combining image preprocessing, scientific metadata, image sampling, and biodiversity-specific workflow requirements.

`Paidiverpy` fills this gap by combining established image-processing tools within a domain-oriented framework for defining, validating, and executing reproducible preprocessing workflows.

# Software Design

`Paidiverpy` was designed around the separation of workflow orchestration from image-processing algorithms. Rather than implementing a new image-processing library, it integrates established tools and exposes their functionality through a common pipeline abstraction. This choice reduces duplication of mature software while allowing preprocessing operations to be represented consistently, combined into reusable workflows, and extended with domain or project-specific algorithms.

The central abstraction is a sequence of processing layers. Operations are grouped into components including `OpenLayer`, `ConvertLayer`, `ColourLayer`, `PositionLayer`, `SamplingLayer`, and `CustomLayer`. A central `Pipeline` coordinates these layers and maintains the dataset as it moves through the workflow. The layer architecture provides a common lifecycle for otherwise heterogeneous operations and makes it possible to inspect, replace, reorder or extend individual stages without rewriting the complete pipeline.

A second design decision is to make the workflow itself a portable research object. Pipelines can be described using YAML configuration files rather than solely through Python code. Parameters and processing order can therefore be reviewed independently of the implementation, version controlled alongside analyses, and reapplied to another dataset. The configuration is validated before execution, reducing the likelihood that differences between runs arise from silently missing or inconsistent parameters.

Metadata are treated as part of the scientific dataset rather than auxiliary files. `Paidiverpy` supports tabular, and iFDO-compatible metadata [@schoening2022fair], allowing contextual information to remain associated with images while preprocessing steps are executed. This is particularly important for biodiversity imagery because variables such as acquisition time, position, camera orientation and sampling conditions may determine which images should be retained or how they should be processed.

Scalability is provided through optional integration with `Dask`, rather than by implementing a bespoke parallel-computing system. The same high-level pipeline abstraction can therefore be used for local execution and parallel processing. This design keeps domain-specific workflow logic within `Paidiverpy` while delegating scheduling and parallel execution to an established ecosystem.

Extensibility was another explicit requirement. The `CustomLayer` mechanism allows researchers to introduce project-specific preprocessing algorithms while preserving the surrounding workflow, configuration and metadata-management infrastructure.

`Paidiverpy` can be used as a Python library, through a command-line interface or web application, and in containerised environments, all using the same underlying processing model. The project is openly developed on GitHub under the Apache-2.0 licence, with documentation on ReadTheDocs, `pip` installation, Docker images, and example datasets and notebooks demonstrating key workflows such as pipeline creation, parallel execution, object-store access, custom algorithms, RAW image processing, and metadata handling.

# Research Impact Statement

`Paidiverpy` is used within marine-imaging research workflows at the National Oceanography Centre and has begun to be adopted beyond its original development team, supporting the preparation and standardisation of imagery for biodiversity analysis and automated classification.

One external application is the DEAL (DEcentrAlised Learning for automated image analysis and biodiversity monitoring) project, led by Plymouth Marine Laboratory. DEAL develops collaborative machine-learning approaches for plankton and seafloor imagery and uses `Paidiverpy` for metadata handling and standardisation within its workflows.

`Paidiverpy` is also being evaluated/used alongside `deep-framex`, an open-source tool for extracting scientific frames from deep-sea video. `deep-framex` produces self-describing frames with embedded metadata and iFDO manifests, providing a complementary workflow in which frame extraction and metadata management precede preprocessing and analysis with `Paidiverpy`.

Within the National Oceanography Centre, `Paidiverpy` is used for **XXXXXXXXXXXXXX**, providing a common workflow for preprocessing **[benthic/plankton/pelagic/etc.]** imagery. These applications have also informed example datasets and best-practice guidance for biodiversity image preprocessing.

The package is intended as reusable infrastructure between image acquisition and downstream manual or automated interpretation, rather than software tied to a single dataset or analysis.

# Conclusion

`Paidiverpy` provides a flexible and extensible framework for preprocessing ecological image datasets. Its layer-based design, explicit workflow definitions, metadata integration, and support for parallel execution address key limitations in existing image-based biodiversity analysis workflows. The package enables users to construct transparent preprocessing pipelines and generate analysis-ready datasets for downstream ecological and AI-based applications.

The software is openly available and distributed with example datasets, pre-built Docker images, and validation tools to support adoption. By standardizing image preprocessing workflows and improving repeatability, `Paidiverpy` contributes a reusable software component for large-scale biodiversity monitoring and ecological research.

# Acknowledgements

This project was supported by the UK Natural Environment Research Council (NERC) through the *Tools for automating image analysis for biodiversity monitoring (AIAB)* Funding Opportunity, reference code **UKRI052**.

# References
