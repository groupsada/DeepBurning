Since the rootfile system image is larger than 100MB which is the maximum file limit allowed by github, we split the compressed image into 50MB partitions. When it is pulled from github, you can use the following command to rebuild the original unified rootfs for booting the FPGA board with linux.
```console 
cat rootfs-* > rootfs.tar.gz
```

