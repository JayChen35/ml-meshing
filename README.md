# ML-Driven Meshing for FEM
My research project for 2024-25: create a machine learning model to find where best to apply mesh refinement for finite element methods (in this case, CFD for airfoils).

## Description
### `meshes/`
run1 is at -5 deg angle of attack
run2 is at -4 deg
...
run16 is at 10 deg angle of attack

### `walldist/`
The files walldistance*.txt contain the wall distances for all elements in a given mesh.  For each element, there are three numbers, which are the wall distances at the three nodes of the triangular element. Yes, there's repetition here, since in a mesh, triangles share nodes, so you'll see the same number repeated 5-6 times for different elements. But this gives you an easy way to obtain the average wall distance for an element: just average the three given wall distances.

## Additional Notes
To use the cluster, do:
1. Activate the U-M VPN (Cisco Secure Client)
2. `ssh pmultigrid.engin.umich.edu`

To copy a directory recursively over SSH, use `rsync` (`scp` is less flexible)
`rsync -avz -e ssh --exclude='.*' source/ user@remote_host:destination/`
- `-a`: This option stands for "archive" and it ensures that all files are copied **recursively** and that all attributes (such as permissions, timestamps, etc.) are preserved.
- `-v`: This option stands for "verbose" and it provides detailed output of the transfer process.
- `-z`: This option enables compression during transfer, which can speed up the process if you are transferring a large amount of data.
- `-e ssh`: This tells rsync to use SSH for the data transfer.
- `source/`: The trailing slash after the directory name is important. It indicates that the contents of the directory source_directory should be copied.
- `--exclude='.*'`: This excludes all files and folders that start with a dot (`.`).
For example, my local command is:
`rsync -avz -e ssh --exclude='.*' ../ml-meshing jasonyc@pmultigrid.engin.umich.edu:~`

To generate dependencies in a `requirements.txt`:
`pip3 install pipreqs`
`pipreqs . --ignore ".venv"`
