use std::collections::HashSet;

use rayon::prelude::*;
use ndarray::prelude::*;

use crate::time_it;

type Embedding = Vec<f32>;
type Scores = Vec<Vec<f32>>;
type Index = usize;
type Score = f32;
type Clusters = Vec<(Index, Vec<Index>)>;
type Community = (Index, Vec<Index>);

const MIN_CLUSTER_SIZE: usize = 5;
const MIN_SIMILARITY: f32 = 0.70;

fn unique_clusters(communities: &Clusters) -> Clusters {
    let mut found: Clusters = Vec::new();
    let mut seen: HashSet<Index> = HashSet::new();

    for (centroid_idx, doc_idxs) in communities.iter() {
        if !doc_idxs.iter().any(|idx| seen.contains(idx)) {
            seen.extend(doc_idxs); // add all doc_idsx to the seen set
            found.push((centroid_idx.to_owned(), doc_idxs.to_owned()));
        }
    }

    found
}

pub fn normalize_all(embeddings: Vec<Embedding>) -> Vec<Embedding> {
    fn norm(a: &[f32]) -> Vec<f32> {
        let z = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        a.iter().map(|x| x / z).collect()
    }

    time_it!(
        "norm",
        let embeddings: Vec<Embedding> = embeddings.iter().map(|v|norm(v)).collect();
    );

    embeddings
}

/// This version uses ndarray for faster matrix multiplication
/// Also optimized the communities stage, python version doesnt actually sort the members in the community
/// It's back to using N^2 memory though, needs to have the pipeline added back...
/// ndarray can probably do the normalization for us too...
pub fn cluster_using_ndarray(embeddings: Vec<Embedding>) -> Clusters {
    let a = Array::from_shape_vec(
        (embeddings.len(), embeddings[0].len()),
        embeddings.clone().into_iter().flatten().collect(),
    )
        .expect("to get dimension correct");

    time_it!("mm",
        let b = a.dot(&a.t());
    );

    time_it!("communities",
        let mut c: Vec<Community> = vec![];
        let mut i = 0;
        for row in b.rows() {
            let count = row.fold(0, |i, v| if *v > MIN_SIMILARITY { i + 1 } else { i });
            if count > MIN_CLUSTER_SIZE {
                c.push(
                    (
                        i,
                        row.indexed_iter()
                            .filter_map(|(i, f)| if f > &MIN_SIMILARITY { Some(i) } else { None })
                            .collect()
                    )
                )
            }
            i = i + 1;
        }
    );

    c.sort_unstable_by(|(_, a), (_, b)| b.len().cmp(&a.len()));

    time_it!("unique",
        let found = unique_clusters(&c);
    );

    found
}


pub fn cluster_using_ndarray_low_memory(embeddings: Vec<Embedding>) -> Clusters {
    // convert list<list<float>> into 2d matrix
    let embeddings = Array::from_shape_vec(
        (embeddings.len(), embeddings[0].len()),
        embeddings.clone().into_iter().flatten().collect(),
    ).expect("to get dimension correct");

    let embeddings_transposed = embeddings.t().clone();

    let mut c: Vec<Community> = embeddings.axis_iter(Axis(0))
        .into_par_iter()
        .enumerate()
        .flat_map(|(doc_index, embedding)| {
            let scores = embedding.dot(&embeddings_transposed);
            let count = scores.fold(0, |initial, score| if *score > MIN_SIMILARITY { initial + 1 } else { initial });
            if count > MIN_CLUSTER_SIZE {
                Some(
                    (
                        doc_index,
                        // doc indexes with scores > MIN_SIMILARITY
                        scores.indexed_iter()
                            .filter_map(|(i, f)| if f > &MIN_SIMILARITY { Some(i) } else { None })
                            .collect()
                    )
                )
            } else {
                None
            }
        })
        .collect();

    // sort by community size
    c.sort_unstable_by(|(_, a), (_, b)| b.len().cmp(&a.len()));

    unique_clusters(&c)
}

pub fn cluster_using_ndarray_batched(embeddings: Vec<Embedding>) -> Clusters {
    // convert list<list<float>> into 2d matrix
    let embeddings = Array::from_shape_vec(
        (embeddings.len(), embeddings[0].len()),
        embeddings.clone().into_iter().flatten().collect(),
    ).expect("to get dimension correct");

    let embeddings_transposed = embeddings.t().clone();

    let mut c: Vec<Community> = vec![];
    let mut i = 0;

    for scores in embeddings
        .axis_chunks_iter(Axis(0), 1000)
        .map(|chunk| chunk.dot(&embeddings_transposed)) {
        for row in scores.rows() {
            let count = row.fold(0, |i, v| if *v > MIN_SIMILARITY { i + 1 } else { i });
            if count > MIN_CLUSTER_SIZE {
                c.push(
                    (
                        i,
                        row.indexed_iter()
                            .filter_map(|(i, f)| if f > &MIN_SIMILARITY { Some(i) } else { None })
                            .collect()
                    )
                )
            }
            i = i + 1;
        }
        drop(scores);
    }

    c.sort_unstable_by(|(_, a), (_, b)| b.len().cmp(&a.len()));

    unique_clusters(&c)
}


pub fn cluster_using_ndarray_batched_unique_on_the_go(embeddings: Vec<Embedding>) -> Clusters {
    // convert list<list<float>> into 2d matrix
    let doc_count = embeddings.len();
    let embeddings = Array::from_shape_vec(
        (doc_count, embeddings[0].len()),
        embeddings.clone().into_iter().flatten().collect(),
    ).expect("to get dimension correct");

    let embeddings_transposed = embeddings.t().clone();

    let mut c: Vec<Community> = vec![];
    let mut i = 0;

    for scores in embeddings
        .axis_chunks_iter(Axis(0), 1000)
        .map(|chunk| chunk.dot(&embeddings_transposed)) {
        for row in scores.rows() {
            let count = row.fold(0, |i, v| if *v > MIN_SIMILARITY { i + 1 } else { i });
            if count > MIN_CLUSTER_SIZE {
                c.push(
                    (
                        i,
                        row.indexed_iter()
                            .filter_map(|(i, f)| if f > &MIN_SIMILARITY { Some(i) } else { None })
                            .collect()
                    )
                )
            }
            i = i + 1;
        }

        drop(scores);

        c.sort_unstable_by(|(_, a), (_, b)| b.len().cmp(&a.len()));

        c = unique_clusters(&c);
    }

    c
}
