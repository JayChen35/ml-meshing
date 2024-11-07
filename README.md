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
