use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::time_it;
use crate::timer::Timer;

pub fn load_text(filename: &str) -> (Vec<Vec<f32>>, Vec<String>) {
    use rust_bert::pipelines::sentence_embeddings::builder::SentenceEmbeddingsBuilder;
    use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModelType;

    time_it!(
        "reading lines of text",
        let file = File::open(filename).unwrap();
        let reader = BufReader::new(file);
        let lines = reader.lines().map(|x| x.unwrap()).collect::<Vec<String>>();
        println!("loaded {} lines", lines.len());
    );

    time_it!(
        "loading sentence_embeddings model",
        // same as we use in arty
        let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2).create_model().unwrap();
    );

    time_it!(
        "sentence_embeddings",
        let embeddings: Vec<Vec<f32>> = lines.chunks(1000).flat_map(|c|model.encode(c).unwrap()).collect();
    );

    (embeddings, lines)
}

pub fn dump_as_json<T>(filename: &str, data: &T)
where
    T: serde::ser::Serialize,
{
    time_it!(
        "dumping json",
        std::fs::write(filename, serde_json::to_string(&data).unwrap()).expect("Failed to write file");
    );
}

pub fn load_vectors_from_json(filename: &str) -> Vec<Vec<f32>> {
    time_it!(
        "reading vectors from json",
        let file = File::open(filename).unwrap();
        let embeddings = serde_json::from_reader(&file).expect("failed to parse json");
    );
    embeddings
}
