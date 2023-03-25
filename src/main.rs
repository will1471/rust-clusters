use clap::{arg, Command};

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

mod cluster;
mod file;
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

        _ => unreachable!(),
    }
}
