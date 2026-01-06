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

Biodiversity is declining rapidly at both global [@diaz2019ipbes] and national scales, including in the UK [@burns2023stateofnature]. Addressing this decline requires robust monitoring to quantify ecosystem change and assess the impacts of human activities, including climate change [@portner2022ipcc]. Consequently, biodiversity monitoring is an international priority, reflected in frameworks such as the Kunming–Montreal Global Biodiversity Framework [@unep2022gbf].

Monitoring is based on repeated measurements of ecosystem attributes such as abundance, biomass, cover, and species richness, which underpin standardized metrics including the Essential Biodiversity Variables [@kissling2018essential]. Large-scale initiatives such as the Global Ocean Observing System [@moltmann2019goos] require data that are comparable across space and time. However, many monitoring approaches remain largely manual, limiting scalability and standardization.

Advances in remote sensing and digital photography now enable large-scale, image-based biodiversity monitoring [@durden2016imagebased]. While these methods offer high sampling efficiency and reduced disturbance, deriving quantitative metrics from images depends on consistent preprocessing and robust metadata integration to ensure comparability across surveys conducted under varying conditions [@durden2016imagebased].

Artificial Intelligence (AI) is increasingly used to extract ecological information from large image datasets [@hoye2021deep]. Although automated detection and classification methods have advanced, less attention has been paid to the preprocessing workflows required to support reliable AI-based analyses [@crosby2023annotation]. The lack of accessible and repeatable preprocessing pipelines that integrate metadata management remains a key barrier to scaling image-based biodiversity monitoring. To be broadly useful, such pipelines must support interoperability, long-term maintenance, and compliance with the FAIR principles [@wilkinson2016fair].

`Paidiverpy` was developed to address this need. It is an open-source Python package that provides a flexible framework for constructing, documenting, and executing preprocessing workflows for biodiversity image analysis. Users define pipelines by combining processing layers, inspect outputs at each stage, and export complete workflow descriptions in standardized formats. Supported operations include image loading, conversion, color correction, resampling, spatial adjustment, and custom user-defined processing. Images are stored as "xarray.Dataset" objects, preserving the association between image data and metadata and enforcing consistent dimensions throughout the workflow. `Paidiverpy` supports multiple metadata standards, optional parallel execution using Dask, and extensibility through custom layers, facilitating integration with downstream AI-based annotation tools.

`Paidiverpy` can be used through Python scripts, a command-line interface, Docker containers, or a web-based graphical user interface. The package includes example datasets, configuration templates, and validation tools to support reproducible use and adoption across a range of ecological imaging applications.

# Statement of Need

Imaging technologies are increasingly used in ecology to observe and quantify natural processes. These approaches generate large volumes of data, motivating the use of AI to automate analysis. Reliable ecological inference from image data, however, depends on consistent and well-documented preprocessing, regardless of whether subsequent analysis is manual or automated. In practice, preprocessing pipelines are often tailored to individual projects, but existing tools present several limitations:

* **Heterogeneous image sources**: Ecological image datasets are collected using diverse platforms and instruments, resulting in varied file formats, metadata structures, and analytical requirements.
* **Inconsistent metadata handling**: Spatial, temporal, and contextual metadata are essential for ecological interpretation, yet limited support for established standards complicates integration and reuse.
* **Complex preprocessing requirements**: Common operations, such as color correction, resampling, orientation adjustment, and backscatter removal, are difficult to combine into coherent workflows using general-purpose libraries.
* **Scalability and repeatability constraints**: Large datasets require efficient parallel processing, consistent dimension handling, and workflows that can be shared and rerun without manual intervention.

General-purpose image processing libraries such as OpenCV [@opencv], scikit-image [@vanderwalt2014skimage], and Pillow [@pillow] provide many low-level operations but do not offer an integrated framework tailored to ecological workflows. As a result, researchers often construct ad hoc preprocessing scripts, which are time-consuming to maintain and difficult to reproduce or share.

`Paidiverpy` addresses these challenges by providing:

1. **Layer-based pipeline architecture**: Discrete preprocessing tasks are implemented as layers (`OpenLayer`, `ConvertLayer`, `ColourLayer`, `PositionLayer`, `SamplingLayer`, `CustomLayer`) and executed through a central **Pipeline** class.
2. **Metadata integration**: Support for iFDO-compliant [@schoenning2022ifdo] JSON and CSV metadata, with automatic extraction and updating of EXIF information.
3. **Explicit workflow definition and validation**: Complete preprocessing workflows are specified using YAML configuration files and validated using built-in tools.
4. **Scalable execution**: Integration with Dask enables multi-threaded and distributed processing, while padding strategies enforce consistent image dimensions.
5. **Multiple user interfaces**: A Python API, command-line interface, Docker images, and a web-based GUI support diverse user workflows.
6. **Example datasets**: Included plankton and benthic image datasets demonstrate typical preprocessing use cases.

`Paidiverpy` does not seek to introduce new image processing algorithms, although it allows users to implement custom processing steps. Instead, it builds on established and well-maintained libraries such as OpenCV, scikit-image, and Pillow, integrating their functionality within a coherent and reproducible preprocessing framework. By focusing on orchestration, metadata handling, and workflow definition rather than algorithm development, `Paidiverpy` enables users to combine existing tools more effectively and apply them consistently across datasets. This approach allows researchers to improve the quality, transparency, and scalability of their analyses without duplicating functionality already provided by mature image processing libraries.

# Conclusion

`Paidiverpy` provides a flexible and extensible framework for preprocessing ecological image datasets. Its layer-based design, explicit workflow definitions, metadata integration, and support for parallel execution address key limitations in existing image-based biodiversity analysis workflows. The package enables users to construct transparent preprocessing pipelines and generate analysis-ready datasets for downstream ecological and AI-based applications.

The software is openly available and distributed with example datasets, pre-built Docker images, and validation tools to support adoption. By standardizing image preprocessing workflows and improving repeatability, `Paidiverpy` contributes a reusable software component for large-scale biodiversity monitoring and ecological research.

# Acknowledgements

This project was supported by the UK Natural Environment Research Council (NERC) through the *Tools for automating image analysis for biodiversity monitoring (AIAB)* Funding Opportunity, reference code **UKRI052**.

# References
