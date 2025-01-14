.. _why:

Why Paidiverpy?
===============

Ecologists are increasingly relying on imaging systems to study and monitor biodiversity. These tools generate massive amounts of data and scientists are looking to Artificial Intelligence (AI) to support alleviate the analysis burden. To effectively use complex AI models to generate biodiversity metrics requires designing bespoke presprocessing pipelines. The tools to do this do not ship with AI packages and ecologists are left to cobble together a workflow from disperate sources. **Paidiverpy** is designed to bridge this gap by providing a streamlined Python package to build preprocessing workflows tailored for biodiversity analysis.

The complexity of biodiversity image datasets can be daunting from a preprocessing perspective:

* Image data come from diverse instruments--each with its own format, metadata, and analysis requirements--necessitating pipelines tailored to each project.
* Different deployment strategies require specific subsampling approaches to ensure unbiased estimates of relevant biodiversity metrics.
* Pixel level preprocessing steps (e.g. colour correction, backscatter removal) can be difficult to piece together from existing Python packages.
* Most existing software toolkits lack the flexibility and comprehensiveness needed for diverse ecological research applications.

Less manual effort, more insightful analysis
--------------------------------------------

**Paidiverpy** aims to simplify the preprocessing of biodiversity image data, enabling researchers to focus on their scientific questions rather than the intricacies of data management. With **Paidiverpy**, users can:

* Easily configure and manage their image processing pipelines through a straightforward interface.
* Visualize every step of their preprocessing pipeline to ensure efficacy 
* Utilize built-in support for several standard metadata formats, ensuring seamless integration with existing datasets.
* Leverage advanced processing features, including parallelization with Dask, to handle large datasets efficiently.
* Enhance reproducability by automatically outputting documentation of their procedure. 

**Paidiverpy** abstracts the complexities of image data preprocessing, allowing you to concentrate on your analysis and research goals.
