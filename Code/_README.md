# Quebec Trees Dataset

This dataset was generated and used in the preprint “Influence of Temperate Forest Autumn Leaf Phenology on Segmentation of Tree Species from UAV Imagery Using Deep Learning”. 

Cloutier, M., Germain, M., & Laliberté, E. (2023). Influence of Temperate Forest Autumn Leaf Phenology on Segmentation of Tree Species from UAV Imagery Using Deep Learning (p. 2023.08.03.548604). bioRxiv. https://doi.org/10.1101/2023.08.03.548604

For rapid visualization of the data:
* [Imagery and annotations](https://arcg.is/1L1DL00) 
* [Point clouds](https://umontreal.maps.arcgis.com/apps/instant/3dviewer/index.html?appid=0d48f1bd9bfc43a5a8e1ca53af02edbe) 


## Abstract

Remote sensing of forests has become increasingly accessible with the use of unoccupied aerial vehicles (UAV), along with deep learning, allowing for repeated high-resolution imagery and the capturing of phenological changes at larger spatial and temporal scales. In temperate forests during autumn, leaf senescence occurs when leaves change colour and drop. However, few UAV-acquired datasets follow the same individual species throughout a growing season at the individual tree level, allowing for a multitude of applications when used with deep learning. Here, we acquired high-resolution UAV imagery over a temperate forest in Quebec, Canada on seven occasions between May and October 2021. We segmented and labeled 23,000 tree crowns from 14 different classes to train and validate a CNN for each imagery acquisition. The dataset includes high-resolution RGB orthomosaics for seven dates in 2021, as well as associated photogrammetric point clouds. The dataset should be useful to develop new algorithms for instance segmentation and species classification of trees from drone imagery.


## Classes

| Label   | Common name         | Scientific name         | Family       | Annotations |
|---------|---------------------|-------------------------|--------------|-------------|
| ABBA    | Balsam fir          | _Abies balsamea_        | Pinaceae     | 2895        |
| ACPE    | Striped maple       | _Acer pensylvanicum_    | Sapindaceae  | 751         |
| ACRU    | Red maple           | _Acer rubrum_           | Sapindaceae  | 5857        |
| ACSA    | Sugar maple         | _Acer saccharum_        | Sapindaceae  | 1014        |
| BEAL    | Yellow birch        | _Betula alleghaniensis_ | Betulaceae   | 290         |
| BEPA    | Paper birch         | _Betula papyrifera_     | Betulaceae   | 5894        |
| FAGR    | American beach      | _Fagus grandifolia_     | Fagaceae     | 222         |
| LALA    | Tamarack            | _Larix laricina_        | Pinaceae     | 185         |
| Picea   | Spruce              | _Picea_ spp.            | Pinaceae     | 1022        |
| PIST    | White pine          | _Pinus strobus_         | Pinaceae     | 569         |
| Populus | Aspen               | _Populus_ spp.          | Salicaceae   | 1114        |
| THOC    | Eastern white cedar | _Thuja occidentalis_    | Cupressaceae | 1510        |
| TSCA    | Eastern hemlock     | _Tsuga canadensis_      | Pinaceae     | 59          |
| Mort    | Dead tree           | -                       | -            | 878         |
| Total   |                     |                         |              | 22,260      |

The genus level classes, _Picea_ spp. and _Populus_ spp., include trees annotated at the species level (PIGL: _Picea glauca_, PIMA: _Picea mariana_, PIRU: _Picea rubens_, POGR: _Populus grandidentata_, POTR: _Populus tremuloides_). These classes were merged due to the difficulty in identifying the species or the similarities between the species.

Not included in this table are approximately 700 additional trees that were segmented and labeled and included in broader categories or in categories with too few individuals.

## Description of the data

This dataset was generated by members of the Plant Functional Ecology Laboratory ([LEFO](https://lefo.ca/)) at the Plant Biology Research Institute ([IRBV](https://irbv.umontreal.ca/)) of Université de Montréal in Quebec, Canada. This includes the imagery acquisition, the field surveys for tree crown labeling, and the segmentation of the tree crowns as well as generating the products (orthomosaics, point clouds).

The imagery was acquired on seven different dates between the end of May and early October in 2021 over a temperate forest at the Station de biologie des Laurentides (SBL) of Université de Montréal in Saint-Hippolyte, Quebec, Canada. The acquisition was done using a DJI Phantom 4 RTK. The processing of the imagery was done using Agisoft Metashape Professional v.1.7.5 to generate the orthomosaics and the point clouds using Structure from motion photogrammetry. Three orthomosaics were generated for each date using a point cloud with medium density and aggressive filtering in order to minimize artifacts. The point clouds in this dataset are high density and have not been filtered for more dense point clouds. The annotations and other vector data were done using ArcGIS Pro v.2.9 and correspond to the individual tree crowns labeled at the species or genus level, or as dead trees (see Table 1 for details).

The detailed methodology as well as additional information on the dataset can be found in this preprint:

Cloutier, M., Germain, M., & Laliberté, E. (2023). Influence of Temperate Forest Autumn Leaf Phenology on Segmentation of Tree Species from UAV Imagery Using Deep Learning (p. 2023.08.03.548604). bioRxiv. https://doi.org/10.1101/2023.08.03.548604

## Data included

The data is organized by acquisition date (YYYY-MM-DD). There are seven acquisition dates and the study site is divided into three zones.

The data included for each of the dates and zones are:
* RGB imagery in Cloud-Optimized GeoTIFF (COG)
* Point cloud in Cloud-Optimized Point Cloud (COPC, .laz files)

The vector layers included are:
* Individual tree level annotations in GeoPackage (GPKG), one for each zone
* Polygons delimiting the inference data used in the publication

A copy of the vector data is in each compressed file for each date.

Metadata files are also included for all the data in a separate folder.
