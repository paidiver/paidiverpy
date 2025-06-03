# Changelog

## [Unreleased](https://github.com/paidiver/paidiverpy/tree/HEAD)

[Full Changelog](https://github.com/paidiver/paidiverpy/compare/v0.1.3...HEAD)

**Implemented enhancements:**

- Add the possibility to work with several images at the same time on the image processing \(instead of one image per time\) [\#200](https://github.com/paidiver/paidiverpy/issues/200)
- Function to open and export metadata to different formats [\#187](https://github.com/paidiver/paidiverpy/issues/187)
- Generate interactive notebook from config [\#184](https://github.com/paidiver/paidiverpy/issues/184)
- Create a configuration file json schema automatically [\#164](https://github.com/paidiver/paidiverpy/issues/164)
- Create an Output Class to Standardize Jupyter Notebook Outputs [\#29](https://github.com/paidiver/paidiverpy/issues/29)
- Illumination correction [\#19](https://github.com/paidiver/paidiverpy/issues/19)
- Addressing blur [\#17](https://github.com/paidiver/paidiverpy/issues/17)
- Contrast alteration  [\#14](https://github.com/paidiver/paidiverpy/issues/14)
- Feature to visualize images in low resolution [\#4](https://github.com/paidiver/paidiverpy/issues/4)

**Fixed bugs:**

- Clean the configuration parser and correct bug with the default params for each step [\#196](https://github.com/paidiver/paidiverpy/issues/196)
- Correct resize function [\#182](https://github.com/paidiver/paidiverpy/issues/182)
- Adjust edge dectection method [\#50](https://github.com/paidiver/paidiverpy/issues/50)

**Closed issues:**

- Reference funding details in readme [\#204](https://github.com/paidiver/paidiverpy/issues/204)
- Raise error is not been added per step \(it is only added in the beginning of the pipeline\) [\#201](https://github.com/paidiver/paidiverpy/issues/201)
- Validate new config added when you add a new step in the pipeline [\#199](https://github.com/paidiver/paidiverpy/issues/199)
- Are resample by depth and by altitude the same thing? [\#198](https://github.com/paidiver/paidiverpy/issues/198)
- Deploy code on bioconda [\#194](https://github.com/paidiver/paidiverpy/issues/194)
- Rename pelagic to plankton [\#188](https://github.com/paidiver/paidiverpy/issues/188)
- Enhance crop function [\#183](https://github.com/paidiver/paidiverpy/issues/183)
- Create a conda distribution and add build for mac and windows [\#174](https://github.com/paidiver/paidiverpy/issues/174)
- Fix and Re-enable SonarCloud Workflow in CI [\#162](https://github.com/paidiver/paidiverpy/issues/162)
- Include metadata in each image preprocessing step [\#156](https://github.com/paidiver/paidiverpy/issues/156)
- Increase test coverage [\#84](https://github.com/paidiver/paidiverpy/issues/84)
- Add data investigation feature [\#26](https://github.com/paidiver/paidiverpy/issues/26)
- Create a simple frontend using streamlit for pipeline management [\#25](https://github.com/paidiver/paidiverpy/issues/25)
- Colour alteration  [\#13](https://github.com/paidiver/paidiverpy/issues/13)
- Add support to load different types of images [\#2](https://github.com/paidiver/paidiverpy/issues/2)

**Merged pull requests:**

- 186 enhance exif handling in processed images [\#210](https://github.com/paidiver/paidiverpy/pull/210) ([soutobias](https://github.com/soutobias))
- 187 function to open and export metadata to different formats [\#208](https://github.com/paidiver/paidiverpy/pull/208) ([soutobias](https://github.com/soutobias))
- 2 add support to load different types of images [\#206](https://github.com/paidiver/paidiverpy/pull/206) ([soutobias](https://github.com/soutobias))
- update config validation [\#203](https://github.com/paidiver/paidiverpy/pull/203) ([soutobias](https://github.com/soutobias))
- 84 increase test coverage [\#202](https://github.com/paidiver/paidiverpy/pull/202) ([soutobias](https://github.com/soutobias))
- update tests [\#197](https://github.com/paidiver/paidiverpy/pull/197) ([soutobias](https://github.com/soutobias))
- 194 deploy code on bioconda [\#195](https://github.com/paidiver/paidiverpy/pull/195) ([soutobias](https://github.com/soutobias))
- 174 create a conda distribution [\#192](https://github.com/paidiver/paidiverpy/pull/192) ([soutobias](https://github.com/soutobias))

## [v0.1.3](https://github.com/paidiver/paidiverpy/tree/v0.1.3) (2025-03-06)

[Full Changelog](https://github.com/paidiver/paidiverpy/compare/v0.1.2...v0.1.3)

## [v0.1.2](https://github.com/paidiver/paidiverpy/tree/v0.1.2) (2025-03-06)

[Full Changelog](https://github.com/paidiver/paidiverpy/compare/v0.1.1...v0.1.2)

**Closed issues:**

- Create a ci/cd that runs the notebook automatically [\#190](https://github.com/paidiver/paidiverpy/issues/190)
- Link Checker Report - 2025-02-26 [\#178](https://github.com/paidiver/paidiverpy/issues/178)
- Link Checker Report - 2025-02-26 [\#177](https://github.com/paidiver/paidiverpy/issues/177)
- Link Checker Report - 2025-02-25 [\#176](https://github.com/paidiver/paidiverpy/issues/176)
- Link Checker Report - 2025-02-24 [\#173](https://github.com/paidiver/paidiverpy/issues/173)
- Link Checker Report - 2025-02-24 [\#172](https://github.com/paidiver/paidiverpy/issues/172)
- image file extension bug \(using pipeline.save\_images\) [\#170](https://github.com/paidiver/paidiverpy/issues/170)
- Run sequential and parallel in custom layer is not standardised with the code [\#163](https://github.com/paidiver/paidiverpy/issues/163)
- Link Checker Report - 2025-02-13 [\#161](https://github.com/paidiver/paidiverpy/issues/161)
- Link Checker Report - 2025-02-13 [\#160](https://github.com/paidiver/paidiverpy/issues/160)
- Update documentation gallery symbols and zenodo link to lastest version [\#158](https://github.com/paidiver/paidiverpy/issues/158)
- Remove double image extension when exporting images [\#157](https://github.com/paidiver/paidiverpy/issues/157)
- Link Checker Report - 2025-02-12 [\#155](https://github.com/paidiver/paidiverpy/issues/155)
- Link Checker Report - 2025-02-12 [\#154](https://github.com/paidiver/paidiverpy/issues/154)
- Link Checker Report - 2025-02-12 [\#152](https://github.com/paidiver/paidiverpy/issues/152)
- Link Checker Report - 2025-02-12 [\#151](https://github.com/paidiver/paidiverpy/issues/151)
- Link Checker Report - 2025-02-12 [\#150](https://github.com/paidiver/paidiverpy/issues/150)
- Link Checker Report - 2025-02-12 [\#149](https://github.com/paidiver/paidiverpy/issues/149)
- Link Checker Report - 2025-02-12 [\#148](https://github.com/paidiver/paidiverpy/issues/148)
- Link Checker Report - 2025-02-12 [\#147](https://github.com/paidiver/paidiverpy/issues/147)
- Link Checker Report - 2025-02-12 [\#146](https://github.com/paidiver/paidiverpy/issues/146)
- Link Checker Report - 2025-02-12 [\#145](https://github.com/paidiver/paidiverpy/issues/145)
- Link Checker Report - 2025-02-12 [\#144](https://github.com/paidiver/paidiverpy/issues/144)
- Link Checker Report - 2025-02-12 [\#143](https://github.com/paidiver/paidiverpy/issues/143)
- Link Checker Report - 2025-02-12 [\#141](https://github.com/paidiver/paidiverpy/issues/141)
- Image converting issue [\#137](https://github.com/paidiver/paidiverpy/issues/137)
- Glossary documentation  [\#22](https://github.com/paidiver/paidiverpy/issues/22)

**Merged pull requests:**

- 163 run sequential and parallel in custom layer is not standardised with the code [\#180](https://github.com/paidiver/paidiverpy/pull/180) ([soutobias](https://github.com/soutobias))
- 158 update documentation gallery symbols and zenodo link to lastest version [\#175](https://github.com/paidiver/paidiverpy/pull/175) ([soutobias](https://github.com/soutobias))
- 157 remove double image extension when exporting images [\#171](https://github.com/paidiver/paidiverpy/pull/171) ([soutobias](https://github.com/soutobias))
- 137 image converting issue [\#159](https://github.com/paidiver/paidiverpy/pull/159) ([Mojtabamsd](https://github.com/Mojtabamsd))

## [v0.1.1](https://github.com/paidiver/paidiverpy/tree/v0.1.1) (2025-02-12)

[Full Changelog](https://github.com/paidiver/paidiverpy/compare/v0.1.0...v0.1.1)

## [v0.1.0](https://github.com/paidiver/paidiverpy/tree/v0.1.0) (2025-02-12)

[Full Changelog](https://github.com/paidiver/paidiverpy/compare/v0.0.2...v0.1.0)

**Closed issues:**

- Link Checker Report - 2025-02-12 [\#140](https://github.com/paidiver/paidiverpy/issues/140)
- Link Checker Report - 2025-02-12 [\#139](https://github.com/paidiver/paidiverpy/issues/139)
- Link Checker Report - 2025-02-12 [\#138](https://github.com/paidiver/paidiverpy/issues/138)
- Link Checker Report - 2025-02-12 [\#135](https://github.com/paidiver/paidiverpy/issues/135)
- Link Checker Report - 2025-02-12 [\#134](https://github.com/paidiver/paidiverpy/issues/134)
- Link Checker Report - 2025-02-11 [\#132](https://github.com/paidiver/paidiverpy/issues/132)
- Link Checker Report - 2025-02-10 [\#131](https://github.com/paidiver/paidiverpy/issues/131)
- Link Checker Report - 2025-02-10 [\#130](https://github.com/paidiver/paidiverpy/issues/130)
- Link Checker Report - 2025-02-10 [\#129](https://github.com/paidiver/paidiverpy/issues/129)
- Link Checker Report - 2025-02-10 [\#128](https://github.com/paidiver/paidiverpy/issues/128)
- Link Checker Report - 2025-02-10 [\#126](https://github.com/paidiver/paidiverpy/issues/126)
- Link Checker Report - 2025-02-10 [\#125](https://github.com/paidiver/paidiverpy/issues/125)
- Link Checker Report - 2025-02-10 [\#124](https://github.com/paidiver/paidiverpy/issues/124)
- Link Checker Report - 2025-02-07 [\#123](https://github.com/paidiver/paidiverpy/issues/123)
- Link Checker Report - 2025-02-07 [\#122](https://github.com/paidiver/paidiverpy/issues/122)
- Link Checker Report - 2025-02-07 [\#120](https://github.com/paidiver/paidiverpy/issues/120)
- Link Checker Report - 2025-02-07 [\#119](https://github.com/paidiver/paidiverpy/issues/119)
- Link Checker Report - 2025-02-07 [\#118](https://github.com/paidiver/paidiverpy/issues/118)
- Link Checker Report - 2025-02-07 [\#117](https://github.com/paidiver/paidiverpy/issues/117)
- Link Checker Report - 2025-02-07 [\#116](https://github.com/paidiver/paidiverpy/issues/116)
- Link Checker Report - 2025-02-07 [\#115](https://github.com/paidiver/paidiverpy/issues/115)
- Correct logs and error handler for methods [\#114](https://github.com/paidiver/paidiverpy/issues/114)
- Link Checker Report - 2025-02-06 [\#113](https://github.com/paidiver/paidiverpy/issues/113)
- Link Checker Report - 2025-02-06 [\#112](https://github.com/paidiver/paidiverpy/issues/112)
- Link Checker Report - 2025-02-05 [\#111](https://github.com/paidiver/paidiverpy/issues/111)
- Link Checker Report - 2025-02-04 [\#109](https://github.com/paidiver/paidiverpy/issues/109)
- Link Checker Report - 2025-02-04 [\#108](https://github.com/paidiver/paidiverpy/issues/108)
- Update documentation [\#106](https://github.com/paidiver/paidiverpy/issues/106)
- Link Checker Report - 2025-02-01 [\#105](https://github.com/paidiver/paidiverpy/issues/105)
- Link Checker Report - 2025-01-30 [\#104](https://github.com/paidiver/paidiverpy/issues/104)
- Code review [\#87](https://github.com/paidiver/paidiverpy/issues/87)
- Create github actions and add runner [\#32](https://github.com/paidiver/paidiverpy/issues/32)
- Lazy load catalog [\#24](https://github.com/paidiver/paidiverpy/issues/24)
- Add feature to work with images in the cloud/lazy load [\#23](https://github.com/paidiver/paidiverpy/issues/23)

**Merged pull requests:**

- 32 create GitHub actions and add runner [\#133](https://github.com/paidiver/paidiverpy/pull/133) ([soutobias](https://github.com/soutobias))
- 32 create GitHub actions and add runner [\#127](https://github.com/paidiver/paidiverpy/pull/127) ([soutobias](https://github.com/soutobias))
- 87 code review [\#110](https://github.com/paidiver/paidiverpy/pull/110) ([soutobias](https://github.com/soutobias))
- 24 lazy load catalog [\#107](https://github.com/paidiver/paidiverpy/pull/107) ([soutobias](https://github.com/soutobias))

## [v0.0.2](https://github.com/paidiver/paidiverpy/tree/v0.0.2) (2025-01-14)

[Full Changelog](https://github.com/paidiver/paidiverpy/compare/v0.0.1...v0.0.2)

**Closed issues:**

- Link Checker Report - 2025-01-13 [\#101](https://github.com/paidiver/paidiverpy/issues/101)
- Link Checker Report - 2025-01-13 [\#99](https://github.com/paidiver/paidiverpy/issues/99)
- Link Checker Report - 2025-01-01 [\#88](https://github.com/paidiver/paidiverpy/issues/88)
- Release the first version of the package on pypi [\#65](https://github.com/paidiver/paidiverpy/issues/65)
- Next step: Citation data [\#42](https://github.com/paidiver/paidiverpy/issues/42)

**Merged pull requests:**

- docs : prepare for the first release [\#103](https://github.com/paidiver/paidiverpy/pull/103) ([soutobias](https://github.com/soutobias))
- feat : correct citation [\#90](https://github.com/paidiver/paidiverpy/pull/90) ([soutobias](https://github.com/soutobias))

## [v0.0.1](https://github.com/paidiver/paidiverpy/tree/v0.0.1) (2025-01-13)

[Full Changelog](https://github.com/paidiver/paidiverpy/compare/43a603ff9e0615cffd59cec8d7b117610f9f332e...v0.0.1)

**Closed issues:**

- Link Checker Report - 2025-01-13 [\#98](https://github.com/paidiver/paidiverpy/issues/98)
- Link Checker Report - 2025-01-13 [\#97](https://github.com/paidiver/paidiverpy/issues/97)
- Link Checker Report - 2025-01-13 [\#96](https://github.com/paidiver/paidiverpy/issues/96)
- Link Checker Report - 2025-01-13 [\#95](https://github.com/paidiver/paidiverpy/issues/95)
- Link Checker Report - 2025-01-13 [\#94](https://github.com/paidiver/paidiverpy/issues/94)
- Link Checker Report - 2025-01-13 [\#93](https://github.com/paidiver/paidiverpy/issues/93)
- Link Checker Report - 2025-01-13 [\#92](https://github.com/paidiver/paidiverpy/issues/92)
- Link Checker Report - 2025-01-13 [\#91](https://github.com/paidiver/paidiverpy/issues/91)
- Create a Marimba and Paidiverpy Integration Example [\#81](https://github.com/paidiver/paidiverpy/issues/81)
- Link Checker Report - 2024-12-01 [\#79](https://github.com/paidiver/paidiverpy/issues/79)
- Correct documentation autoapi [\#77](https://github.com/paidiver/paidiverpy/issues/77)
- Correct json schema to be non case sensitive [\#72](https://github.com/paidiver/paidiverpy/issues/72)
- user guide documentation from best practice [\#70](https://github.com/paidiver/paidiverpy/issues/70)
- Link Checker Report - 2024-11-01 [\#66](https://github.com/paidiver/paidiverpy/issues/66)
- Allow Users to Integrate Custom Algorithms into the pipeline [\#63](https://github.com/paidiver/paidiverpy/issues/63)
- Create a config schema, config validator and a website for creating config files [\#57](https://github.com/paidiver/paidiverpy/issues/57)
- Link Checker Report - 2024-10-01 [\#56](https://github.com/paidiver/paidiverpy/issues/56)
- Create  sample dataset methods to download sample data [\#53](https://github.com/paidiver/paidiverpy/issues/53)
- Test IFDO file as metadata file [\#51](https://github.com/paidiver/paidiverpy/issues/51)
- Next step: Sonarcloud integration [\#46](https://github.com/paidiver/paidiverpy/issues/46)
- Next step: Enable Zenodo integration [\#45](https://github.com/paidiver/paidiverpy/issues/45)
- Next step: Read the Docs [\#44](https://github.com/paidiver/paidiverpy/issues/44)
- Next step: Linting [\#43](https://github.com/paidiver/paidiverpy/issues/43)
- Create an example pipeline for benthos images [\#41](https://github.com/paidiver/paidiverpy/issues/41)
- Create an example pipeline for pelagic [\#40](https://github.com/paidiver/paidiverpy/issues/40)
- Next step: Sonarcloud integration [\#39](https://github.com/paidiver/paidiverpy/issues/39)
- Next step: Enable Zenodo integration [\#38](https://github.com/paidiver/paidiverpy/issues/38)
- Next step: Read the Docs [\#36](https://github.com/paidiver/paidiverpy/issues/36)
- Next step: Linting [\#35](https://github.com/paidiver/paidiverpy/issues/35)
- Next step: Citation data [\#34](https://github.com/paidiver/paidiverpy/issues/34)
- Create documentation [\#33](https://github.com/paidiver/paidiverpy/issues/33)
- Create first tests [\#31](https://github.com/paidiver/paidiverpy/issues/31)
- Add docstring to functions [\#30](https://github.com/paidiver/paidiverpy/issues/30)
- Add parallelisation to the code [\#28](https://github.com/paidiver/paidiverpy/issues/28)
- Rename catalog name?  [\#21](https://github.com/paidiver/paidiverpy/issues/21)
- British or USA english? [\#20](https://github.com/paidiver/paidiverpy/issues/20)
- Vignette removal  [\#16](https://github.com/paidiver/paidiverpy/issues/16)
- Create a rename step [\#3](https://github.com/paidiver/paidiverpy/issues/3)

**Merged pull requests:**

- style: correct styling" [\#86](https://github.com/paidiver/paidiverpy/pull/86) ([soutobias](https://github.com/soutobias))
- style: correct names [\#85](https://github.com/paidiver/paidiverpy/pull/85) ([soutobias](https://github.com/soutobias))
- 31 create first tests [\#83](https://github.com/paidiver/paidiverpy/pull/83) ([soutobias](https://github.com/soutobias))
- 64 add slurm job submission for pipeline runs on hpc systems [\#82](https://github.com/paidiver/paidiverpy/pull/82) ([soutobias](https://github.com/soutobias))
- 77 correct documentation autoapi [\#80](https://github.com/paidiver/paidiverpy/pull/80) ([soutobias](https://github.com/soutobias))
- feat: add environment.yml fiile and update docs [\#1](https://github.com/paidiver/paidiverpy/pull/1) ([soutobias](https://github.com/soutobias))



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
