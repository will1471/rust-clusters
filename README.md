Convert a file of newline seperated documents into vectors.

```
$ cargo run --release --features dhat-heap -- vectors lines.txt clusters.json
   Compiling cluster v0.1.0 (/home/will/src/github.com/will1471/cluster)
    Finished release [optimized + debuginfo] target(s) in 9.43s
     Running `target/release/cluster vectors lines.txt clusters.json`
2023-03-23 18:07:42.787389895 +00:00 Started reading lines of text
loaded 10000 lines
2023-03-23 18:07:42.809433054 +00:00 Finished reading lines of text, took: PT0.022043159S
2023-03-23 18:07:42.809438830 +00:00 Started loading sentence_embeddings model
2023-03-23 18:07:44.772968156 +00:00 Finished loading sentence_embeddings model, took: PT1.963529326S
2023-03-23 18:07:44.773020216 +00:00 Started sentence_embeddings
2023-03-23 18:08:38.675774912 +00:00 Finished sentence_embeddings, took: PT53.902754696S
2023-03-23 18:08:38.698585578 +00:00 Started dumping json
2023-03-23 18:08:38.863283012 +00:00 Finished dumping json, took: PT0.164697434S
dhat: Total:     402,933,318 bytes in 2,760,934 blocks
dhat: At t-gmax: 82,894,457 bytes in 10,219 blocks
dhat: At t-end:  184,102 bytes in 197 blocks
dhat: The data has been saved to dhat-heap.json, and is viewable with dhat/dh_view.html
```

Cluster vectors using algo 1, used 427 MB peak

```
$ cargo run --release --features dhat-heap -- cluster-stages clusters.json clusters1.json
    Finished release [optimized + debuginfo] target(s) in 0.11s
     Running `target/release/cluster cluster-stages clusters.json clusters1.json`
2023-03-23 18:09:32.512517597 +00:00 Started reading vectors from json
2023-03-23 18:10:06.940515936 +00:00 Finished reading vectors from json, took: PT34.427998339S
2023-03-23 18:10:06.940534532 +00:00 Started norm
2023-03-23 18:10:06.968999605 +00:00 Finished norm, took: PT0.028465073S
2023-03-23 18:10:06.970179783 +00:00 Started scores
2023-03-23 18:10:11.686217195 +00:00 Finished scores, took: PT4.716037412S
2023-03-23 18:10:11.686250523 +00:00 Started communities
2023-03-23 18:10:14.266898841 +00:00 Finished communities, took: PT2.580648318S
2023-03-23 18:10:14.266919721 +00:00 Started unique
2023-03-23 18:10:14.267969680 +00:00 Finished unique, took: PT0.001049959S
2023-03-23 18:10:14.280845574 +00:00 Started dumping json
2023-03-23 18:10:14.281042142 +00:00 Finished dumping json, took: PT0.000196568S
dhat: Total:     2,882,953,829 bytes in 190,783 blocks
dhat: At t-gmax: 427,401,220 bytes in 30,134 blocks
dhat: At t-end:  65,320 bytes in 106 blocks
dhat: The data has been saved to dhat-heap.json, and is viewable with dhat/dh_view.html
```

Cluster vectors using algo 2, used 38 MB peak

```
$ cargo run --release --features dhat-heap -- cluster-merged clusters.json clusters2.json
    Finished release [optimized + debuginfo] target(s) in 0.11s
     Running `target/release/cluster cluster-merged clusters.json clusters2.json`
2023-03-23 18:11:37.020473370 +00:00 Started reading vectors from json
2023-03-23 18:12:11.655680546 +00:00 Finished reading vectors from json, took: PT34.635207176S
2023-03-23 18:12:11.655699150 +00:00 Started norm
2023-03-23 18:12:11.683603702 +00:00 Finished norm, took: PT0.027904552S
2023-03-23 18:12:11.684792970 +00:00 Started comb
2023-03-23 18:12:17.044870088 +00:00 Started unique
2023-03-23 18:12:17.045980230 +00:00 Finished unique, took: PT0.001110142S
2023-03-23 18:12:17.045987842 +00:00 Finished comb, took: PT5.361194872S
2023-03-23 18:12:17.049840295 +00:00 Started dumping json
2023-03-23 18:12:17.050001145 +00:00 Finished dumping json, took: PT0.000160850S
dhat: Total:     2,882,975,117 bytes in 194,539 blocks
dhat: At t-gmax: 36,479,748 bytes in 20,025 blocks
dhat: At t-end:  65,320 bytes in 106 blocks
dhat: The data has been saved to dhat-heap.json, and is viewable with dhat/dh_view.html
```

Todo, use binary format instead of json? Check the clustering still works? Try a matrix library...


## Matrix Library

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
