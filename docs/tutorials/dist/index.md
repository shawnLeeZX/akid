# Distributed akid

In this tutorial, we will write a program that does computation distributedly.
The full code, `dist_akid` can be found under `example` folder.

OK, now it is just some pointers.

We use a `Source`, a `Sensor`, a `Brain` in this case. Read
[How To](../../how_tos/index.html) to know what they do. Also read the
[tensorflow tutorial](https://www.tensorflow.org/versions/r0.11/how_tos/distributed/index.html#distributed-tensorflow)
for distributed computation. This tutorial ports the distributed example
provided by tensorflow. The usage of the program is the same as in the
tensorflow tutorial.

After successfully running the program, you are supposed to see outputs like:

```bash
2.45249
2.40535
2.29056
2.2965
2.25567
2.27914
2.26652
2.27446
2.2911
2.26182
2.17706
2.18829
2.23567
2.21965
2.20997
2.14844
2.10352
2.066
2.12029
2.10526
2.10102
2.03739
2.04613
2.05246
2.04463
2.03297
```

which is the training loss.
