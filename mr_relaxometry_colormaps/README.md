# mr_relaxometry_colormaps

As recommended in Fuderer et al 2024 (https://doi.org/10.1002/mrm.30290), use Lapari for T1/R1 maps and Navia for T2/R2 maps. 

These maps are not currently in 3D Slicer. Code has been written in convert_colormap_slicer.py to create color tables (https://slicer.readthedocs.io/en/latest/developer_guide/modules/colors.html) to be used in Slicer.

The code also has a section to remap the colormap to "log-like" curve, adapted from github.com/mfuderer/colorResources/blob/main/colorLogRemap.m