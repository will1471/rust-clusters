use clap::{arg, Command};
use crate::phatic::PhaticDetectorBuilder;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

mod cluster;
mod file;
mod phatic;
mod timer;

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
}

macro_rules! get_arg {
    ($matches:expr, $id:literal) => {
        $matches
            .get_one::<String>($id)
            .map(|s| s.as_str())
            .expect("Expected arg...")
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

            let (e, _) = file::load_text(input);
            file::dump_as_json(output, &e);
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
            let embeddings = cluster::normalize_all(embeddings);
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
            let embeddings = cluster::normalize_all(embeddings);
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
            let embeddings = cluster::normalize_all(embeddings);
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
            let embeddings = cluster::normalize_all(embeddings);
            time_it!(
                "main_cluster",
                let clusters = cluster::cluster_using_ndarray_batched_unique_on_the_go(embeddings);
            );
            file::dump_as_json(output, &clusters);
        }

        _ => unreachable!(),
    }
}
