# Theory

Our current clustering algorithm works basically as follows:

We have a set of vectors, of equal length, we want to cluster.

We're going to compare 10,000 vectors, of length 384.

We put the data into a 2d matrix. (384 x 10000 x 4 bytes) ~ 15 MB

```
  384
+-----+
|  I  |
|     |  10,000
|     |
|     |
|     |
|     |
+-----+
```

We run a normalization function over the values (inplace).

We transpose the matrix, (10000 x 384 x 4 bytes) ~ 15 MB

```
        10,000
+---------------------+
| I't                 |  384
+---------------------+
```

We do a matrix multiply of the two (10000 x 10000 x 4bytes) ~ 381 MB

```
                  10,000
          +---------------------+
          | I't                 |  384
          +---------------------+
  384
+-----+   +---------------------+
|  I  |   |                     |
|     |   |                     |  10,000
|     |   |                     |
|     |   |                     |
|     |   |                     |
|     |   |                     |
+-----+   +---------------------+
```

We iterate the large matrix row by row, creating a list of "Communities". Where a Community
is set of vector indices close to the input vector.

After collecting all Communities, we sort them, largest to smallest.

We filter the Communities such that they have unique members. ie we always have the first (largest) Community,
then we have any communities where we've not seen the vector indices before. This requires a Set that grows
upto 10,000 items.

## Observation 1

We don't need the entire 10,000 x 10,000 matrix at the same time. We consume it row by row once.

Instead of allocating 10,000 x 10,000 matrix, we could allocate a 1 x 10,000 matrix.

## Observation 2

In the pathological case, where each vector is equal, the collection of all Communities could grow to 
10,000 x 10,000.
