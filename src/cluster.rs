use std::collections::HashSet;

use rayon::prelude::*;

use crate::time_it;
use crate::timer::Timer;

type Clusters = Vec<(usize, Vec<usize>)>;
type Embedding = Vec<f32>;
type Scores = Vec<Vec<f32>>;

const MIN_CLUSTER_SIZE: usize = 3;
const MIN_SIMILARITY: f32 = 0.60;

fn dot(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(a, b)| a * b).sum()
}

fn communities(scores: &Scores) -> Clusters {
    let mut communities: Clusters = Vec::new();

    for (i, v) in scores.iter().enumerate() {
        let mut sorted = v
            .iter()
            .enumerate()
            .map(|i| (i.0, *i.1))
            .collect::<Vec<(usize, f32)>>();
        sorted.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        if sorted[MIN_CLUSTER_SIZE - 1].1 > MIN_SIMILARITY {
            communities.push((
                i,
                sorted
                    .iter()
                    .take_while(|(_, v)| *v > MIN_SIMILARITY)
                    .map(|(i, _)| *i)
                    .collect(),
            ));
        }
    }
    communities.par_sort_by(|(_, a), (_, b)| b.len().cmp(&a.len()));
    communities
}

fn unique_clusters(communities: &Clusters) -> Clusters {
    let mut found: Clusters = Vec::new();
    let mut seen: HashSet<usize> = HashSet::new();
    for (centroid_idx, doc_idxs) in communities.iter() {
        if !doc_idxs.iter().any(|idx| seen.contains(idx)) {
            seen.extend(doc_idxs); // add all doc_idsx to the seen set
            found.push((*centroid_idx, doc_idxs.clone()));
        }
    }
    found
}

pub fn cluster_using_discrete_stages(embeddings: Vec<Embedding>) -> Clusters {
    time_it!(
    "scores",
    let scores: Vec<Vec<f32>> = embeddings
        .par_iter()
        .map(|v1| embeddings.iter().map(|v2| dot(v1, v2)).collect::<Vec<f32>>())
        .collect();
    );

    time_it!(
        "communities",
        let communities = communities(&scores);
    );

    time_it!(
        "unique",
        let found = unique_clusters(&communities);
    );

    found
}

pub fn cluster_using_combined_pipeline(embeddings: Vec<Embedding>) -> Clusters {
    fn scores_to_community(i: usize, embedding_scores: Vec<f32>) -> Option<(usize, Vec<usize>)> {
        let mut sorted = embedding_scores
            .iter()
            .enumerate()
            .map(|i| (i.0, *i.1))
            .collect::<Vec<(usize, f32)>>();

        sorted.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        if sorted[MIN_CLUSTER_SIZE - 1].1 > MIN_SIMILARITY {
            Some((
                i,
                sorted
                    .iter()
                    .take_while(|(_, v)| *v > MIN_SIMILARITY)
                    .map(|p| p.0)
                    .collect(),
            ))
        } else {
            None
        }
    }

    time_it!("comb",
        let mut communities = embeddings
            .par_iter()
            .enumerate()
            .filter_map(
                |(idx, embedding)| scores_to_community(
                    idx,
                    // this is basically a matrix multiply with one embedding * other embeddings,
                    // would it be more efficient to have a matrix multiply function instead of
                    // calling dot many times?
                    embeddings.iter()
                        .map(|other_embedding| dot(embedding, other_embedding))
                        .collect::<Vec<f32>>()
                )
            )
            .collect::<Vec<(usize,Vec<usize>)>>();

        communities.par_sort_by(|(_, a), (_, b)| b.len().cmp(&a.len()));

        time_it!(
            "unique",
            let found = unique_clusters(&communities);
        );
    );
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
