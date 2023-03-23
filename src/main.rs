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
            Command::new("cluster-stages")
                .about("Read a file of vectors, dump a file of clusters")
                .arg(arg!(<VECTOR_FILE> "input file"))
                .arg(arg!(<CLUSTER_FILE> "outfile file")),
        )
        .subcommand(
            Command::new("cluster-merged")
                .about("Read a file of vectors, dump a file of clusters")
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

        Some(("cluster-stages", submatch)) => {
            let input = get_arg!(submatch, "VECTOR_FILE");
            let output = get_arg!(submatch, "CLUSTER_FILE");

            let embeddings = file::load_vectors_from_json(input);
            let embeddings = cluster::normalize_all(embeddings);
            let clusters = cluster::cluster_using_discrete_stages(embeddings);
            file::dump_as_json(output, &clusters);
        }

        Some(("cluster-merged", submatch)) => {
            let input = get_arg!(submatch, "VECTOR_FILE");
            let output = get_arg!(submatch, "CLUSTER_FILE");

            let embeddings = file::load_vectors_from_json(input);
            let embeddings = cluster::normalize_all(embeddings);
            let clusters = cluster::cluster_using_combined_pipeline(embeddings);
            file::dump_as_json(output, &clusters);
        }

        _ => unreachable!(),
    }
}
