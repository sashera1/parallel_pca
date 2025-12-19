To run: first, run makefile by navigating to parallel_pca and running makefile

Then, run ./pca <relative path to input matrix> <dimensions> [-d] 
dimensions is how many pca dimensions you want returned
-d (or -debug) optional flag for debug mode

This program only runs on a square integer matrix of the format:

rowCount
colCount
<row of integers, each separated by a space>
<next row>
...
<final row>