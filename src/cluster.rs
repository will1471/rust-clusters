use std::collections::HashSet;
use std::ops::IndexMut;

use rayon::prelude::*;
use ndarray::prelude::*;

use crate::time_it;
use crate::timer::Timer;

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

fn push_clusters(scores: &Array2<f32>, communities: &mut Vec<Community>, index_offset: usize) {
    let mut i = index_offset;
    for row in scores.rows() {
        if count_scores_over_threshold(&row) > MIN_CLUSTER_SIZE {
            communities.push((i, idx_over_threshold(&row).iter().map(|i| i + index_offset).collect()));
        }
        i = i + 1;
    }
}

/// This version uses ndarray for faster matrix multiplication
/// Also optimized the communities stage, python version doesnt actually sort the members in the community
/// It's back to using N^2 memory though, needs to have the pipeline added back...
/// ndarray can probably do the normalization for us too...
pub fn cluster_using_ndarray(embeddings: Vec<Embedding>) -> Clusters {
    let a = vectors_to_array(embeddings);
    cluster_no_splitting(&a, 0)
}

pub fn cluster_no_splitting(embeddings: &Array2<f32>, index_offset: usize) -> Clusters {
    cluster_no_splitting_view(&embeddings.view(), index_offset)
}

pub fn cluster_no_splitting_view(embeddings: &ArrayView2<f32>, index_offset: usize) -> Clusters {
    let mut c: Vec<Community> = vec![];
    push_clusters(&embeddings.dot(&embeddings.t()), &mut c, index_offset);
    c.sort_unstable_by(|(_, a), (_, b)| b.len().cmp(&a.len()));
    unique_clusters(&c)
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
    const CHUNK_SIZE: usize = 1000;

    for scores in embeddings
        .axis_chunks_iter(Axis(0), CHUNK_SIZE)
        .map(|chunk| chunk.dot(&embeddings_transposed))
    {
        push_clusters(&scores, &mut c, i * CHUNK_SIZE);
        i = i + 1;
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
    const CHUNK_SIZE: usize = 1000;

    for scores in embeddings
        .axis_chunks_iter(Axis(0), CHUNK_SIZE)
        .map(|chunk| chunk.dot(&embeddings_transposed))
    {
        push_clusters(&scores, &mut c, CHUNK_SIZE * i);
        i = i + 1;
        c.sort_unstable_by(|(_, a), (_, b)| b.len().cmp(&a.len()));
        c = unique_clusters(&c);
    }

    c
}

pub fn cluster_massive(embeddings: Vec<Embedding>) -> Clusters {
    let embeddings = vectors_to_array(embeddings);

    let mut global_communities: Vec<Community> = vec![];
    let mut i = 0;
    const CHUNK_SIZE: usize = 1000;

    for chunk in embeddings
        .axis_chunks_iter(Axis(0), CHUNK_SIZE)
    {
        let local_communities = cluster_no_splitting_view(&chunk, CHUNK_SIZE * i);
        let local_communities_len = local_communities.len();
        let local_communities_doc_count = count_docs_in_clusters(&local_communities);

        i = i + 1;

        global_communities = merge_clusters(global_communities, local_communities, &embeddings);

        println!(
            "Chunk {} - found {} clusters, with {} docs in chunk. After merge: {} clusters, {} docs",
            i,
            local_communities_len,
            local_communities_doc_count,
            global_communities.len(),
            count_docs_in_clusters(&global_communities)
        );
    }

    let t = Timer::start("Fitting all docs into found clusters...");

    let cluster_embeddings = get_embeddings(&centroids(&global_communities), &embeddings);

    let seen_docs: HashSet<usize> = clustered_document_indexes(&global_communities);

    let mut doc_index = 0;
    for doc in embeddings.rows() {
        if !seen_docs.contains(&doc_index) {
            let scores = doc.dot(&cluster_embeddings.t());

            let mut scores_i: Vec<(usize, f32)> = Vec::with_capacity(scores.len());
            for s in scores.indexed_iter() {
                scores_i.push((s.0, *s.1));
            }
            scores_i.sort_unstable_by(|(_, a), (_, b)| b.partial_cmp(&a).unwrap());
            if scores_i[0].1 > MIN_SIMILARITY {
                global_communities[scores_i[0].0].1.push(doc_index);
            }
        }
        doc_index = doc_index + 1;
    }

    t.end();

    println!(
        "Total {} clusters, {} docs",
        global_communities.len(),
        count_docs_in_clusters(&global_communities)
    );

    for c1 in &global_communities {
        for c2 in &global_communities {
            if c1.0 == c2.0 {
                continue;
            }
            let c1_e = get_embeddings(&[c1.0], &embeddings);
            let c2_e = get_embeddings(&[c2.0], &embeddings);
            let score = c1_e.row(0).dot(&c2_e.row(0).t());
            if score > MIN_SIMILARITY {
                println!("score {} too close? {}, {}", score, c1.0, c2.0);
                assert!(false);
            }
        }
    }

    global_communities
}

fn centroids(clusters: &Clusters) -> Vec<Index> {
    clusters.iter().map(|(centroid, _)| *centroid).collect()
}

fn get_embeddings(idx: &[usize], embeddings: &Array2<f32>) -> Array2<f32> {
    let mut v = Vec::with_capacity(idx.len());

    for i in idx {
        v.push(embeddings.slice(s![*i,..]).to_vec());
    }
    vectors_to_array(v)
}

fn merge_clusters(a: Clusters, b: Clusters, embeddings: &Array2<f32>) -> Clusters {
    if a.len() == 0 {
        return b;
    }

    if b.len() == 0 {
        return a;
    }

    let a_embeddings = get_embeddings(&centroids(&a), &embeddings);
    let b_embeddings = get_embeddings(&centroids(&b), &embeddings);
    let scores = a_embeddings.dot(&b_embeddings.t());
    drop(a_embeddings);
    drop(b_embeddings);

    let mut a_mut = a.clone();

    for (row_index, row) in scores.rows().into_iter().enumerate()
    {
        let row_scores:Vec<(usize,f32)> = row.to_vec().into_iter().enumerate().collect();

        // if row_scores.len() >= 2 {
        //     assert!(row_scores.get(0).unwrap().1 > row_scores.get(1).unwrap().1);
        // }

        for (index, score) in row_scores.iter()
        {
            if *score > MIN_SIMILARITY  {
                // merge b into a
                let merged = merge_community(&a_mut[row_index], &b[*index], &embeddings);
                a_mut[row_index] = merged;
            }
        }
    }

    let mut c: Clusters = vec![];

    for a in a_mut {
        c.push(a);
    }

    for b in b {
        c.push(b);
    }

    let mut c: Clusters = c.into_iter().filter(|(_, m)| m.len() > MIN_CLUSTER_SIZE).collect();
    c.sort_unstable_by(|(_, a), (_, b)| b.len().cmp(&a.len()));
    unique_clusters(&c)
}

fn merge_community(a: &Community, b: &Community, embeddings: &Array2<f32>) -> Community {
    let mut embeddings_in_communities: Vec<Embedding> = Vec::with_capacity(a.1.len() + b.1.len());
    let mut idx: Vec<Index> = Vec::with_capacity(a.1.len() + b.1.len());

    for index in &a.1 {
        embeddings_in_communities.push(embeddings.slice(s![*index,..]).to_vec());
        idx.push(*index);
    }

    for index in &b.1 {
        embeddings_in_communities.push(embeddings.slice(s![*index,..]).to_vec());
        idx.push(*index);
    }

    assert!(embeddings_in_communities.len() > 0);

    let clusters = cluster_using_ndarray(embeddings_in_communities);
    let centroid = clusters[0].0;
    let centroid = idx[centroid];

    let mut members = Vec::new();
    for i in &clusters[0].1 {
        let centroid_embedding = embeddings.slice(s![centroid,..]);
        let member_embedding = embeddings.slice(s![*i,..]);

        if centroid_embedding.dot(&member_embedding.t()) > MIN_SIMILARITY {
            members.push(idx[*i]);
        }
    }

    (centroid, members)
}

fn count_docs_in_clusters(clusters: &Clusters) -> usize {
    let mut i = 0;
    for c in clusters {
        i = i + c.1.len();
    }
    i
}

fn clustered_document_indexes(clusters: &Clusters) -> HashSet<usize> {
    let mut seen = HashSet::new();
    for c in clusters {
        for i in &c.1 {
            seen.insert(*i);
        }
    }
    seen
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
