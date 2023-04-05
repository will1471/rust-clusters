use std::collections::HashSet;

use rayon::prelude::*;
use ndarray::prelude::*;

use crate::time_it;

type Embedding = Vec<f32>;
type Index = usize;
pub type Clusters = Vec<(Index, Vec<Index>)>;
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

pub fn normalize_all_inplace(mut embeddings: Vec<Embedding>) -> Vec<Embedding> {
    fn norm(a: &mut Embedding) {
        let z = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        a.iter_mut().for_each(|x| *x = *x / z);
    }

    time_it!(
        "norm inplace",
        embeddings.iter_mut().for_each(|v|norm(v));
    );
    embeddings
}

fn count_scores_over_threshold(row: &ArrayView1<f32>) -> usize {
    row.fold(0, |i, v| if *v > MIN_SIMILARITY { i + 1 } else { i })
}

fn idx_over_threshold(row: &ArrayView1<f32>) -> Vec<usize> {
    row.indexed_iter()
        .filter_map(|(i, f)| if f > &MIN_SIMILARITY { Some(i) } else { None })
        .collect()
}

pub fn vectors_to_array(embeddings: Vec<Embedding>) -> Array2<f32> {
    let embeddings = Array2::from_shape_vec(
        (embeddings.len(), embeddings[0].len()),
        embeddings.into_iter().flatten().collect(),
    ).expect("to get dimension correct");
    embeddings
}

/// This version uses ndarray for faster matrix multiplication
/// Also optimized the communities stage, python version doesnt actually sort the members in the community
/// It's back to using N^2 memory though, needs to have the pipeline added back...
/// ndarray can probably do the normalization for us too...
pub fn cluster_using_ndarray(embeddings: Vec<Embedding>) -> Clusters {
    let a = vectors_to_array(embeddings);

    time_it!("mm",
        let b = a.dot(&a.t());
    );

    time_it!("communities",
        let mut c: Vec<Community> = vec![];
        let mut i = 0;
        for row in b.rows() {
            if count_scores_over_threshold(&row) > MIN_CLUSTER_SIZE {
                c.push((i, idx_over_threshold(&row)));
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
    let embeddings = vectors_to_array(embeddings);
    let embeddings_transposed = embeddings.t().clone();

    let mut c: Vec<Community> = embeddings.axis_iter(Axis(0))
        .into_par_iter()
        .enumerate()
        .flat_map(|(doc_index, embedding)| {
            let scores = embedding.dot(&embeddings_transposed);
            if count_scores_over_threshold(&scores.view()) > MIN_CLUSTER_SIZE {
                Some((doc_index, idx_over_threshold(&scores.view())))
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
    let embeddings = vectors_to_array(embeddings);

    let embeddings_transposed = embeddings.t().clone();

    let mut c: Vec<Community> = vec![];
    let mut i = 0;

    for scores in embeddings
        .axis_chunks_iter(Axis(0), 1000)
        .map(|chunk| chunk.dot(&embeddings_transposed)) {
        for row in scores.rows() {
            if count_scores_over_threshold(&row) > MIN_CLUSTER_SIZE {
                c.push((i, idx_over_threshold(&row)));
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
    let embeddings = vectors_to_array(embeddings);
    let embeddings_transposed = embeddings.t().clone();

    let mut c: Vec<Community> = vec![];
    let mut i = 0;

    for scores in embeddings
        .axis_chunks_iter(Axis(0), 1000)
        .map(|chunk| chunk.dot(&embeddings_transposed)) {
        for row in scores.rows() {
            if count_scores_over_threshold(&row) > MIN_CLUSTER_SIZE {
                c.push((i, idx_over_threshold(&row)));
            }
            i = i + 1;
        }

        drop(scores);

        c.sort_unstable_by(|(_, a), (_, b)| b.len().cmp(&a.len()));

        c = unique_clusters(&c);
    }

    c
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_can_count_scores_over_threshold() {
        let a = array![1.0, 0.0, 0.1, 2.0, 0.2, 0.0, 0.0];
        assert_eq!(2, count_scores_over_threshold(&a.view()));

        let a = array![];
        assert_eq!(0, count_scores_over_threshold(&a.view()));

        let a = array![0.0, 0.0, 0.1];
        assert_eq!(0, count_scores_over_threshold(&a.view()));
    }

    #[test]
    fn test_it_can_collect_idx_over_threshold() {
        let a = array![];
        assert_eq!(Vec::<usize>::new(), idx_over_threshold(&a.view()));

        let a = array![0.0, 0.0, 0.1];
        assert_eq!(Vec::<usize>::new(), idx_over_threshold(&a.view()));

        let a = array![1.0, 0.0, 0.1, 2.0, 0.2, 0.0, 0.0];
        assert_eq!(vec![0 as usize, 3 as usize], idx_over_threshold(&a.view()));
    }

    #[test]
    fn test_it_can_normalize_vectors() {
        fn vec_f32_compare(a: &[f32], b: &[f32]) -> bool {
            a.iter()
                .zip(b)
                .all(|(a, b)| (a - b).abs() < 0.0001)
        }

        let input = vec![
            vec![0.4, 1.0, -0.3],
            vec![0.2, 0.1, -0.1],
        ];

        let norm = normalize_all_inplace(input);

        let expect = vec![
            vec![0.3577708763, 0.8944271908, -0.2683281572],
            vec![0.8164965809, 0.4082482904, -0.4082482904],
        ];

        assert!(vec_f32_compare(&expect[0], &norm[0]));
        assert!(vec_f32_compare(&expect[1], &norm[1]));

        let norm2 = normalize_all_inplace(norm);

        assert!(vec_f32_compare(&expect[0], &norm2[0]));
        assert!(vec_f32_compare(&expect[1], &norm2[1]));
    }
}
