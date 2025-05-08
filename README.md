DeMagicEye
==========

Reverse's magic eye images into depth maps.


Dependencies (Python 3)
============

- numpy
- opencv


Usage
=====

To create a depth map from an autostereogram, simply pass it as an argument the DeMagicEye.py script, along with a file name stub for output

Usage:

    DeMagicEye <infile> <outfile_stem> [search_window]

Example:

    python DeMagicEye.py autostereograms/face.gif face_output

You will see two output images generated at <outfile_stem>RAW.png and <outfile_stem>Equalized.png

