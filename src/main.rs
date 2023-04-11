use std::ffi::OsStr;
use std::path::Path;
use clap::{arg, Command};
use crate::phatic::PhaticDetectorBuilder;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

mod cluster;
mod file;
mod phatic;
mod timer;
mod tsne;

fn cli() -> Command {
    Command::new("cluster")
        .about("testing clustering")
        .subcommand_required(true)
        .subcommand(
            Command::new("vectors")
                .about("Read a file of text, dump a file of vectors")
                .arg(arg!(<TEXT_FILE> "input file"))
                .arg(arg!(<VECTOR_FILE> "outfile file")),
        )
        .subcommand(
            Command::new("phatic")
                .about("Is a string phatic?")
                .arg(arg!(<INPUT> "input string"))
                .arg(arg!(--similarity <SIMILARITY> "similarity"))
                .arg(arg!(--prevector "give vector to phatic detector?")),
        )
        .subcommand(
            Command::new("cluster-ndarray")
                .about("Read a file of vectors, dump a file of clusters\nUses N^2 memory, optimized")
                .arg(arg!(<VECTOR_FILE> "input file"))
                .arg(arg!(<CLUSTER_FILE> "outfile file")),
        )
        .subcommand(
            Command::new("cluster-ndarray2")
                .about("Read a file of vectors, dump a file of clusters\nUses low memory, optimized")
                .arg(arg!(<VECTOR_FILE> "input file"))
                .arg(arg!(<CLUSTER_FILE> "outfile file")),
        )
        .subcommand(
            Command::new("cluster-ndarray3")
                .about("Read a file of vectors, dump a file of clusters\nUses low memory, optimized")
                .arg(arg!(<VECTOR_FILE> "input file"))
                .arg(arg!(<CLUSTER_FILE> "outfile file")),
        )
        .subcommand(
            Command::new("cluster-ndarray4")
                .about("Read a file of vectors, dump a file of clusters\nUses low memory, unique otg")
                .arg(arg!(<VECTOR_FILE> "input file"))
                .arg(arg!(<CLUSTER_FILE> "outfile file")),
        )
        .subcommand(
            Command::new("cluster-massive")
                .about("Cluster a massive file using ideas from https://ntropy.com/post/clustering-millions-of-sentences-to-optimize-the-ml-workflow")
                .arg(arg!(<VECTOR_FILE> "input file"))
                .arg(arg!(<CLUSTER_FILE> "outfile file"))
                .arg(arg!(-c <COUNT> "Optionally sets the number of vectors to include")),
        )
        .subcommand(
            Command::new("tsne")
                .about("Do a clustering, and use tsne to reduce dimensions")
                .arg(arg!(<VECTOR_FILE> "input file"))
                .arg(arg!(<TSNE_FILE> "output file"))
        )
}

macro_rules! get_arg {
    ($matches:expr, $id:literal) => {
        $matches
            .get_one::<String>($id)
            .map(|s| s.as_str())
            .expect("Expected arg...")
    };
}

macro_rules! get_arg_opt {
    ($matches:expr, $id:literal) => {
        $matches
            .get_one::<String>($id)
            .map(|s| s.as_str())
    };
}

fn main() {
    #[cfg(feature = "dhat-heap")]
        let _profiler = dhat::Profiler::new_heap();

    let matches = cli().get_matches();

    match matches.subcommand() {
        Some(("vectors", submatch)) => {
            let input = get_arg!(submatch, "TEXT_FILE");
            let output = get_arg!(submatch, "VECTOR_FILE");

            let ext = Path::new(output).extension().and_then(OsStr::to_str);

            let write = match ext {
                Some("json") => file::dump_as_json,
                Some("msgpack") => file::dump_as_msgpack,
                Some(ext) => panic!("Dont know how to handle extension: {}", ext),
                None => panic!("Failed to parse extension")
            };

            let (e, _) = file::load_text(input);

            write(output, &e);
        }

        Some(("phatic", submatch)) => {
            let input = get_arg!(submatch, "INPUT");

            let similarity = submatch.get_one::<String>("similarity").expect("expected similarity");
            let similarity = similarity.parse::<f32>().expect("Invalid float");
            if similarity > 0.999 {
                panic!("Similarity too high");
            }
            if similarity < 0.001 {
                panic!("Similarity too low");
            }

            let p = PhaticDetectorBuilder::new()
                .with_similarity_threshold(similarity)
                .build()
                .expect("Expect to construct phatic detector");

            let container;
            let v = match submatch.get_one::<bool>("prevector") {
                Some(true) => {
                    use rust_bert::pipelines::sentence_embeddings::builder::SentenceEmbeddingsBuilder;
                    use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModelType;
                    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2).create_model().unwrap();
                    container = model.encode(&[input]).unwrap();
                    Some(&container[0])
                },
                Some(false) => None,
                None => panic!("Expected prevector flag")
            };

            if p.is_phatic(input, &v).expect("to check is phatic") {
                println!("String is phatic");
            } else {
                println!("String is NOT phatic");
            }
        }

        Some(("cluster-ndarray", submatch)) => {
            let input = get_arg!(submatch, "VECTOR_FILE");
            let output = get_arg!(submatch, "CLUSTER_FILE");

            let embeddings = file::load_vectors_from_json(input);
            let embeddings = cluster::normalize_all_inplace(embeddings);
            time_it!(
                "main_cluster",
                let clusters = cluster::cluster_using_ndarray(embeddings);
            );
            file::dump_as_json(output, &clusters);
        }

        Some(("cluster-ndarray2", submatch)) => {
            let input = get_arg!(submatch, "VECTOR_FILE");
            let output = get_arg!(submatch, "CLUSTER_FILE");

            let embeddings = file::load_vectors_from_json(input);
            let embeddings = cluster::normalize_all_inplace(embeddings);
            time_it!(
                "main_cluster",
                let clusters = cluster::cluster_using_ndarray_low_memory(embeddings);
            );
            file::dump_as_json(output, &clusters);
        }

        Some(("cluster-ndarray3", submatch)) => {
            let input = get_arg!(submatch, "VECTOR_FILE");
            let output = get_arg!(submatch, "CLUSTER_FILE");

            let embeddings = file::load_vectors_from_json(input);
            let embeddings = cluster::normalize_all_inplace(embeddings);
            time_it!(
                "main_cluster",
                let clusters = cluster::cluster_using_ndarray_batched(embeddings);
            );
            file::dump_as_json(output, &clusters);
        }

        Some(("cluster-ndarray4", submatch)) => {
            let input = get_arg!(submatch, "VECTOR_FILE");
            let output = get_arg!(submatch, "CLUSTER_FILE");

            let embeddings = file::load_vectors_from_json(input);
            let embeddings = cluster::normalize_all_inplace(embeddings);
            time_it!(
                "main_cluster",
                let clusters = cluster::cluster_using_ndarray_batched_unique_on_the_go(embeddings);
            );
            file::dump_as_json(output, &clusters);
        }

        Some(("cluster-massive", submatch)) => {
            let input = get_arg!(submatch, "VECTOR_FILE");
            let output = get_arg!(submatch, "CLUSTER_FILE");
            let ext = Path::new(input).extension().and_then(OsStr::to_str);

            let read = match ext {
                Some("json") => file::load_vectors_from_json,
                Some("msgpack") => file::load_vectors_from_msgpack,
                Some(ext) => panic!("Dont know how to handle extension: {}", ext),
                None => panic!("Failed to parse extension")
            };

            let mut embeddings = read(input);

            if let Some(c) = get_arg_opt!(submatch, "COUNT").and_then(|s| s.parse().ok()) {
                embeddings.truncate(c);
            }

            let embeddings = cluster::normalize_all_inplace(embeddings); // 1000000 × 384 × (4 × byte) = 1.536 GB

            println!("count = {}", embeddings.len());

            time_it!(
                "main_cluster",
                let clusters = cluster::cluster_massive(embeddings);
            );
            file::dump_as_json(output, &clusters);
        }

        Some(("tsne", submatch)) => {
            let input = get_arg!(submatch, "VECTOR_FILE");
            let output = get_arg!(submatch, "TSNE_FILE");

            let embeddings = file::load_vectors_from_json(input);
            let embeddings = cluster::normalize_all_inplace(embeddings);

            let embeddings_copy = embeddings.clone();

            time_it!(
                "cluster",
                let clusters = cluster::cluster_using_ndarray_batched(embeddings_copy);
            );

            let embeddings = cluster::vectors_to_array(embeddings);
            time_it!(
                "tsne",
                let reduced = tsne::reduce_dimensions(&clusters, &embeddings);
            );
            file::dump_as_json(output, &reduced);
        }

        _ => unreachable!(),
    }
}
