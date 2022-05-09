
# Overview

This code calculates the Jaccard similarity of vertices in a graph.

# Instructions

To compile:

```
make
```

To run:

```
./jaccard [flags]

```

Optional flags:

```
  -g <graphFile>    name of the file with the graph to solve

  -0                run GPU version 0
  -1                run GPU version 1
  -2                run GPU version 2
  -3                run GPU version 3
                    NOTE: It is okay to specify multiple different GPU versions in the
                          same run. By default, only the CPU version is run.

  -v                perform exact verification of the GPU run across the CPU run

```

