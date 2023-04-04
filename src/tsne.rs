use bhtsne::tSNE;
use ndarray::{Array2, s};

use crate::cluster::Clusters;

pub fn reduce_dimensions(clusters: &Clusters, embeddings: &Array2<f32>) -> Vec<(f32, f32)>
{
    let mut v: Vec<(f32, f32)> = Vec::new();
    if clusters.len() == 0 {
        return v;
    }
    if clusters.len() == 1 {
        v.push((0.0, 0.0));
        return v;
    }
    if clusters.len() == 2 {
        v.push((100.0, 100.0));
        v.push((-100.0, -100.0));
        return v;
    }

    /*
    We have to setup perplexity correctly, otherwise there's some code in the lib that panics.
    I'm not sure what the param does, I'm just satisfying this constraint:

    pub(super) fn check_perplexity<T: Float + AsPrimitive<usize>>(perplexity: &T, n_samples: &usize) {
        if n_samples - 1 < 3 * perplexity.as_() {
            panic!("error: the provided perplexity is too large for the number of data points.\n");
        }
    }
    */

    let mut perplexity: usize = 40;
    while clusters.len() - 1 < 3 * perplexity {
        perplexity = perplexity - 1;
    }

    // get the embeddings for the cluster centroids
    let vectors: Vec<Vec<f32>> = clusters.iter()
        .map(|(idx, _)| embeddings.slice(s![*idx, ..]).clone().to_vec())
        .collect();

    let mut tsne: tSNE<f32, Vec<f32>> = tSNE::new(&vectors);
    let e = tsne.embedding_dim(2)
        .perplexity(perplexity as f32)
        .epochs(2000)
        .exact(|a, b| {
            a.iter()
                .zip(b.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt()
        })
        .embedding();

    // take the x, y pairs from tsne and add the cluster size (useful for graphing)
    let points: Vec<(f32, f32, usize)> = e
        .chunks_exact(2)
        .enumerate()
        .map(|(idx, pair)| (pair[0], pair[1], clusters[idx].1.len()))
        .collect();

    /*
    Code that spits out an image, adding the library brings in many dependancies and slows the build down,
    use features to being in smaller number of dependencies, change lib, or do something else.


    use plotters::backend::BitMapBackend;
    use plotters::chart::ChartBuilder;

    let (x_min, x_max, y_min, y_max) = points
        .iter()
        .fold(
            (0.0, 0.0, 0.0, 0.0),
            |(x_min, x_max, y_min, y_max), (x, y, _)|
                (
                    if x_min < *x { x_min } else { *x },
                    if x_max > *x { x_max } else { *x },
                    if y_min < *y { y_min } else { *y },
                    if y_max > *y { y_max } else { *y }
                ),
        );

    use plotters::prelude::*;

    let root = BitMapBackend::new("output.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE).expect("to be able to fill chart with white pixels");

    let mut scatter = ChartBuilder::on(&root).build_cartesian_2d(x_min..x_max, y_min..y_max).unwrap();
    scatter.draw_series(
        points.iter().map(|(x, y, s)| Circle::new((*x, *y), *s as i32, BLACK.stroke_width(1)))
    ).unwrap();
    root.present().expect("to be able to present the chart");

    */

    points.iter().map(|t| (t.0, t.1)).collect()
}