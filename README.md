# Memory Usage

```
$ bash memory-usage.sh
cluster-ndarray  - 1k vectors = 8.297439 megabytes
cluster-ndarray  - 5k vectors = 121.57206 megabytes
cluster-ndarray  - 10k vectors = 441.95498 megabytes
cluster-ndarray  - 20k vectors = 1.6827208 gigabytes
cluster-ndarray  - 30k vectors = 3.6983211 gigabytes
cluster-ndarray  - 40k vectors = 6.565422 gigabytes

cluster-ndarray2 - 1k vectors = 3.91296 megabytes
cluster-ndarray2 - 5k vectors = 21.897984 megabytes
cluster-ndarray2 - 10k vectors = 43.789442 megabytes
cluster-ndarray2 - 20k vectors = 87.572354 megabytes
cluster-ndarray2 - 30k vectors = 118.77235 megabytes
cluster-ndarray2 - 40k vectors = 175.13818 megabytes

cluster-ndarray3 - 1k vectors = 8.29744 megabytes
cluster-ndarray3 - 5k vectors = 41.597664 megabytes
cluster-ndarray3 - 10k vectors = 82.085538 megabytes
cluster-ndarray3 - 20k vectors = 163.16829 megabytes
cluster-ndarray3 - 30k vectors = 219.37949 megabytes
cluster-ndarray3 - 40k vectors = 326.40087 megabytes
```

# Time

```
$ bash time.sh

cluster-ndarray  -  1k =  0.006146 seconds avg 3 runs (0.006186200 0.005779600 0.006471300)
cluster-ndarray  -  5k =  0.115    seconds avg 3 runs (0.109891100 0.106352200 0.128901200)
cluster-ndarray  - 10k =  0.4568   seconds avg 3 runs (0.471051700 0.462445900 0.436903600)
cluster-ndarray  - 20k =  2.074    seconds avg 3 runs (1.978195100 2.116270100 2.129029100)
cluster-ndarray  - 30k =  5.085    seconds avg 3 runs (5.031557100 5.117108200 5.106502700)
cluster-ndarray  - 40k =  9.251    seconds avg 3 runs (10.061635700 8.980553500 8.711575100)

cluster-ndarray2 -  1k =  0.02414  seconds avg 3 runs (0.023307700 0.024878800 0.024247800)
cluster-ndarray2 -  5k =  0.6683   seconds avg 3 runs (0.653129300 0.667661800 0.684142100)
cluster-ndarray2 - 10k =  2.941    seconds avg 3 runs (3.055077500 2.917979100 2.849310)
cluster-ndarray2 - 20k = 10.66     seconds avg 3 runs (10.638300700 10.662921800 10.666122900)
cluster-ndarray2 - 30k = 24.81     seconds avg 3 runs (24.658328400 24.845688800 24.926793100)
cluster-ndarray2 - 40k = 45.73     seconds avg 3 runs (45.301796600 45.664831200 46.219050900)

cluster-ndarray3 -  1k = 0.006763  seconds avg 3 runs (0.006555100 0.007503400 0.006230200)
cluster-ndarray3 -  5k = 0.1092    seconds avg 3 runs (0.107013300 0.107218300 0.113346700)
cluster-ndarray3 - 10k = 0.4391    seconds avg 3 runs (0.449009200 0.437610500 0.430601)
cluster-ndarray3 - 20k = 1.86      seconds avg 3 runs (1.862618500 1.878791100 1.837911)
cluster-ndarray3 - 30k = 4.275     seconds avg 3 runs (4.219159800 4.299801100 4.306989700)
cluster-ndarray3 - 40k = 7.683     seconds avg 3 runs (7.750780400 7.655981 7.642433100)
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
