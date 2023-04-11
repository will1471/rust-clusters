use std::time::Duration;
use criterion::*;
use cluster::file::*;
use cluster::cluster::*;

fn cluster_bench(c: &mut Criterion) {
    let embeddings = load_vectors_from_msgpack("./data/tweets.msgpack");

    let mut group = c.benchmark_group("cluster");
    group.sample_size(10);
    //group.measurement_time(Duration::from_secs(120));

    for size in [1000, 5000, 10000, 20000, 40000].iter() {
        let mut e = embeddings.clone();
        e.truncate(*size as usize);
        let e = normalize_all_inplace(e);

        group.bench_with_input(BenchmarkId::new("cluster_using_ndarray", size), size, |b, &_size| {
            b.iter(|| cluster_using_ndarray(e.clone()));
        });

        group.bench_with_input(BenchmarkId::new("cluster_using_ndarray_low_memory", size), size, |b, &_size| {
            b.iter(|| cluster_using_ndarray_low_memory(e.clone()));
        });

        group.bench_with_input(BenchmarkId::new("cluster_using_ndarray_batched", size), size, |b, &_size| {
            b.iter(|| cluster_using_ndarray_batched(e.clone()));
        });

        group.bench_with_input(BenchmarkId::new("cluster_using_ndarray_batched_unique_on_the_go", size), size, |b, &_size| {
            b.iter(|| cluster_using_ndarray_batched_unique_on_the_go(e.clone()));
        });
    }

    group.finish();
}

criterion_group!(benches, cluster_bench);
criterion_main!(benches);