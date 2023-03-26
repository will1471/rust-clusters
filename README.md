# Memory Usage

```
$ bash memory-usage.sh
cluster-ndarray  -  1k vectors =   8.297439  megabytes
cluster-ndarray  -  5k vectors = 121.57206   megabytes
cluster-ndarray  - 10k vectors = 441.95498   megabytes
cluster-ndarray  - 20k vectors =   1.6827208 gigabytes
cluster-ndarray  - 30k vectors =   3.6983211 gigabytes
cluster-ndarray  - 40k vectors =   6.565422  gigabytes

cluster-ndarray2 -  1k vectors =   3.91296  megabytes
cluster-ndarray2 -  5k vectors =  21.897984 megabytes
cluster-ndarray2 - 10k vectors =  43.789442 megabytes
cluster-ndarray2 - 20k vectors =  87.572354 megabytes
cluster-ndarray2 - 30k vectors = 118.77235  megabytes
cluster-ndarray2 - 40k vectors = 175.13818  megabytes

cluster-ndarray3 -  1k vectors =   8.29744  megabytes
cluster-ndarray3 -  5k vectors =  41.597664 megabytes
cluster-ndarray3 - 10k vectors =  82.085538 megabytes
cluster-ndarray3 - 20k vectors = 163.16829  megabytes
cluster-ndarray3 - 30k vectors = 219.37949  megabytes
cluster-ndarray3 - 40k vectors = 326.40087  megabytes

cluster-ndarray4 -  1k vectors =   8.29744  megabytes
cluster-ndarray4 -  5k vectors =  41.574688 megabytes
cluster-ndarray4 - 10k vectors =  81.96137  megabytes
cluster-ndarray4 - 20k vectors = 162.73614  megabytes
cluster-ndarray4 - 30k vectors = 218.34823  megabytes
cluster-ndarray4 - 40k vectors = 324.28985  megabytes
```

# Time

Note some variance in run to run times is causing issues in average, need to run more and discard outliers. 

Would help to not run on a laptop that will thermal throttle...

```
$ bash time.sh

cluster-ndarray -  1k = 0.006011 seconds avg 3 runs (0.005894800 0.006161600 0.005976900)
cluster-ndarray -  5k = 0.15 seconds avg 3 runs (0.134679500 0.165467800 0.149927600)
cluster-ndarray - 10k = 0.5458 seconds avg 3 runs (0.543352 0.590569100 0.503415)
cluster-ndarray - 20k = 2.208 seconds avg 3 runs (2.321868800 2.101717500 2.201210700)
cluster-ndarray - 30k = 5.113 seconds avg 3 runs (5.335912600 4.999812700 5.001968500)
cluster-ndarray - 40k = 20.34 seconds avg 3 runs (39.668714800 10.328017100 11.031651200)

cluster-ndarray2 -  1k = 0.02729 seconds avg 3 runs (0.025344200 0.027940 0.028587900)
cluster-ndarray2 -  5k = 0.7555 seconds avg 3 runs (0.765210600 0.682225800 0.819212900)
cluster-ndarray2 - 10k = 3.217 seconds avg 3 runs (3.416109100 3.348304400 2.886095200)
cluster-ndarray2 - 20k = 11.34 seconds avg 3 runs (11.753761600 11.200172800 11.077001700)
cluster-ndarray2 - 30k = 25.39 seconds avg 3 runs (25.235181600 25.343407700 25.594127)
cluster-ndarray2 - 40k = 45.78 seconds avg 3 runs (45.771240300 45.743821200 45.830221200)

cluster-ndarray3 -  1k = 0.007323 seconds avg 3 runs (0.006140700 0.009674400 0.006154400)
cluster-ndarray3 -  5k = 0.1163 seconds avg 3 runs (0.123985800 0.110045500 0.114841800)
cluster-ndarray3 - 10k = 0.5039 seconds avg 3 runs (0.477560700 0.551024500 0.483139900)
cluster-ndarray3 - 20k = 1.929 seconds avg 3 runs (1.890591500 1.912888800 1.983728800)
cluster-ndarray3 - 30k = 6.181 seconds avg 3 runs (4.381790700 5.930157700 8.232483400)
cluster-ndarray3 - 40k = 14.52 seconds avg 3 runs (14.572001300 14.717263100 14.283751200)

cluster-ndarray4 -  1k = 0.01008 seconds avg 3 runs (0.009167 0.011955500 0.009110100)
cluster-ndarray4 -  5k = 0.2302 seconds avg 3 runs (0.224306600 0.218231700 0.247920)
cluster-ndarray4 - 10k = 0.8606 seconds avg 3 runs (0.765741800 0.940686 0.875295700)
cluster-ndarray4 - 20k = 3.578 seconds avg 3 runs (3.622595700 3.518956800 3.592120100)
cluster-ndarray4 - 30k = 8.027 seconds avg 3 runs (8.048342100 8.011991600 8.021660300)
cluster-ndarray4 - 40k = 13.99 seconds avg 3 runs (14.470479200 14.477544700 13.016570700)
```

## Matrix Library (cluster-ndarray)

Manged to write a faster N^2 version using ndarray, and some optimizations in the communities extraction.

```
$ cargo run --release -- cluster-stages 10k.json /dev/null | grep "Finished main_cluster"
2023-03-24 21:03:59.770409 +00:00 Finished main_cluster, took: PT11.427306S

$ cargo run --release -- cluster-ndarray 10k.json /dev/null 2>&1 | grep "Finished main"
2023-03-24 21:37:40.618573300 +00:00 Finished main_cluster, took: PT0.592992S
```

Memory usage of faster one higher because it copies the data into a matrix.

```
$ cargo run --release --features dhat-heap -- cluster-stages 10k.json /dev/null 2>&1 | grep gmax
dhat: At t-gmax: 416,107,080 bytes in 20,130 blocks

$ cargo run --release --features dhat-heap -- cluster-ndarray 10k.json /dev/null 2>&1 | grep gmax
dhat: At t-gmax: 441,886,465 bytes in 10,027 blocks
```

Todo, make the low memory version this fast...

I've just realised, rust-bert package is actually bringing in binding for torch (https://docs.rs/crate/tch/latest),
so I should have access to all the functions we're using to make the python version fast.

## Matrix Library, Low Memory Impl. (cluster-ndarray2)

On 40k vectors, memory usage decreases from 6,261 MB to 167 MB.

```
$ cargo run --release --features dhat-heap -- cluster-ndarray 40k.json /dev/null 2>&1 | grep max
dhat: At t-gmax: 6,565,421,969 bytes in 46,081 blocks

$ cargo run --release --features dhat-heap -- cluster-ndarray2 40k.json /dev/null 2>&1 | grep max
dhat: At t-gmax: 175,138,178 bytes in 47,258 blocks
```

On laptop, 40k vectors, runtime increases from 11.6 seconds to 50.5 seconds for low memory version.

```
$ cargo run --release -- cluster-ndarray2 40k.json /dev/null 2>&1 | grep "Finished main"
2023-03-25 11:41:31.977950700 +00:00 Finished main_cluster, took: PT49.510539600S
2023-03-25 11:42:26.482299100 +00:00 Finished main_cluster, took: PT50.112657900S
2023-03-25 11:43:23.468459300 +00:00 Finished main_cluster, took: PT52.070877S

$ cargo run --release -- cluster-ndarray 40k.json /dev/null 2>&1 | grep "Finished main"
2023-03-25 11:43:43.504264700 +00:00 Finished main_cluster, took: PT12.809869900S
2023-03-25 11:43:58.909638900 +00:00 Finished main_cluster, took: PT11.248993900S
2023-03-25 11:44:14.163945300 +00:00 Finished main_cluster, took: PT11.035947500S
```

On laptop, 10k vectors, runtime increases from 0.59 seconds to 2.83 seconds for low memory version.

```
$ cargo run --release -- cluster-ndarray 10k.json /dev/null 2>&1 | grep "Finished main"
2023-03-25 11:48:19.041532800 +00:00 Finished main_cluster, took: PT0.539656600S
2023-03-25 11:48:21.863955600 +00:00 Finished main_cluster, took: PT0.668131600S
2023-03-25 11:48:23.881252900 +00:00 Finished main_cluster, took: PT0.590317600S

$ cargo run --release -- cluster-ndarray2 10k.json /dev/null 2>&1 | grep "Finished main"
2023-03-25 11:48:31.916033200 +00:00 Finished main_cluster, took: PT2.888060900S
2023-03-25 11:48:36.301082700 +00:00 Finished main_cluster, took: PT2.893791200S
2023-03-25 11:48:47.696331900 +00:00 Finished main_cluster, took: PT2.964115500S
```

To reach these speeds, it's fully utilizing `11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz` in WSL2.

I assume this overhead is from going from one large matrix multiply operation, to 10k or 40k smaller operations.

Maybe the overhead can be reduced by allowing the program to MM several vectors at a time, increasing the memory usage,
but by a tunable amount. (Letting us set a target memory usage.) Any server with enough compute to do the maths
is going to have more than 167MB available to use...

It's also possible that I'm doing something stupid in the memory efficient version, but I'm not experienced
enough with rust / ndarray / rayon to know.

# Batched Matrix Multiplications (cluster-ndarray3)

Implemented hybrid of one big MM, and lots of small MM, a few medium MM (1k vectors at a time.)

Memory growth appears to be linear, appears to run faster than one large MM, not sure why... Maybe because it's easier
to get access to smaller blocks of memory, blocks of memory can be reused?

# Batched Matrix Multiplications, with periodic pruning of Communities (cluster-ndarray4)

In the pathological case, (all vectors equal or very close,) the size of collected communities can grow to 
~ `num_docs^2 * 8byte`.

This impl. prunes the collected communities after each matrix multiplication batch.

This appears to have a significant impact on runtime, maybe it would be a useful protection when combined with a
size check before pruning.

I've not verified the communities grows in the way I expect, and if it does, that the puring works. To verify this
running a memory usage check with a input dataset with equal vectors should show it.

