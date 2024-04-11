# build and run

```
# clone this repository and its submodules
$ git clone --recurse-submodules https://github.com/w3ntao/smallpt-megakernel.git

# build
$ cd smallpt-megakernel
$ mkdir build; cd build
$ cmake ..; make -j

# render
$ ./smallpt-megakernel
rendering (64 spp) took 11.214 seconds.
image saved to `smallpt_megakernel_64.png`
```
