.. _science_guide1:

Position/size-based processing
==============================

Position/size-based processing in biodiversity imaging deals with challenges that result from the spatial and dimensional characteristics of images. The major challenge can emanate from the varying positions, orientations, and sizes of the captured subjects, effects from the imaging equipment, and even environmental conditions during image acquisition. This section focuses on processing techniques that maintain accurate representation of spatial features and dimensions in images, which is key to reliable analysis and interpretation. There exist several unique challenges to processing images:

- **Geometric Distortion**: The changes in angle of the camera, altitude, or lens properties might give misleading measurements and interpretations of actual spatial representations.

- **Lens Distortion**: The optical anomaly in a camera lens, including barrel and pincushion distortions or other distortions, will distort the effective size and/or shape of objects in the image.

- **Overlapping Images:**: In case there are several overlapping images, it needs to be processed delicately so that redundancy can be minimized and, simultaneously, continuity across the image group be ensured.


Removal of images based on waypoints / location
-----------------------------------------------

Filtering out images that do not align with predefined geographic coordinates or waypoints that are relevant to the study area. Predefined geographic coordinates or waypoints are established based on the study area or the specific locations of interest. These waypoints act as reference points for determining which images should be retained or discarded. If an image's location data does not fall within a certain threshold distance from any waypoint, it is flagged for removal.


Removal of images based on altitude, pitch/roll/yaw, vehicle motion/crabbing
----------------------------------------------------------------------------

The removal of images that are captured under suboptimal conditions related to the altitude, orientation (pitch, roll, yaw), and motion (e.g., crabbing).
