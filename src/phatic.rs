use lazy_static::lazy_static;
use regex::Regex;
use std::error::Error;
use ndarray::prelude::*;

use rust_bert::pipelines::sentence_embeddings::{
    Embedding,
    SentenceEmbeddingsModel,
    SentenceEmbeddingsModelType,
    builder::SentenceEmbeddingsBuilder,
};

pub struct PhaticDetector {
    model: SentenceEmbeddingsModel,
    embeddings: Array<f32, Ix2>,
    similarity: f32,
}

static EXAMPLES: &str = include_str!("phatic_examples.txt");

impl PhaticDetector {
    fn new(similarity: f32) -> Result<PhaticDetector, Box<dyn Error>> {
        let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
            .create_model()?;

        let mut embeddings: Vec<Embedding> = vec![];
        for line in EXAMPLES.lines() {
            embeddings.extend(model.encode(&[line])?);
        }
        embeddings = normalize_all(embeddings);

        let embeddings = Array::from_shape_vec(
            (embeddings.len(), embeddings[0].len()),
            embeddings.into_iter().flatten().collect(),
        )?.reversed_axes();

        Ok(PhaticDetector { model, embeddings, similarity })
    }

    pub fn is_phatic(&self, text: &str, embedding: &Option<&Embedding>) -> Result<bool, Box<dyn Error>> {
        match sanitise_text(text).split(char::is_whitespace).count() {
            0..=3 => Ok(true),
            15.. => Ok(false),
            _ => vector_check(text, embedding, self)
        }
    }
}

pub struct PhaticDetectorBuilder {
    similarity_threshold: f32,
}

impl PhaticDetectorBuilder {
    pub fn new() -> PhaticDetectorBuilder {
        PhaticDetectorBuilder { similarity_threshold: 0.5 }
    }

    pub fn with_similarity_threshold(mut self, threshold: f32) -> PhaticDetectorBuilder {
        self.similarity_threshold = threshold;
        self
    }

    pub fn build(self) -> Result<PhaticDetector, Box<dyn Error>> {
        PhaticDetector::new(self.similarity_threshold)
    }
}


fn norm(a: &[f32]) -> Embedding {
    let z = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    a.iter().map(|x| x / z).collect()
}

fn normalize_all(embeddings: Vec<Embedding>) -> Vec<Embedding> {
    embeddings.iter().map(|v| norm(v)).collect()
}

fn vector_check(text: &str, embedding: &Option<&Embedding>, p: &PhaticDetector) -> Result<bool, Box<dyn Error>>
{
    let container: Vec<Embedding>;
    let embedding = if embedding.is_some() {
        embedding.unwrap()
    } else {
        container = normalize_all(p.model.encode(&[text])?);
        &container[0]
    };

    let v = Array::from_shape_vec((1, embedding.len()), embedding.clone().into_iter().collect())?;
    let scores = v.dot(&p.embeddings);
    let count = scores.fold(0, |i, v| if *v > p.similarity { i + 1 } else { i });
    Ok(count > 0)
}

fn sanitise_text(text: &str) -> String {
    // @todo is this correct? we're applying remove_user_mentions twice, and not removing other things...
    let text = remove_user_mentions(text);
    let text = remove_emoticons(&text);
    remove_user_mentions(&text)
}

fn clean_spaces(text: &str) -> String {
    lazy_static! {
        static ref RE: Regex = Regex::new(r" +")
            .expect("To have valid regex");
    }
    RE.replace_all(text, " ").trim().to_owned()
}

// fn remove_new_lines(text: &str) -> String {
//     todo!();
// }

fn remove_user_mentions(text: &str) -> String {
    lazy_static! {
        static ref RE: Regex = Regex::new(r"@[A-Za-z0-9]+")
            .expect("To have valid regex");
    }
    clean_spaces(RE.replace_all(text, "").as_ref())
}

fn remove_emoticons(text: &str) -> String {
    lazy_static! {
        static ref RE: Regex = Regex::new(
            r"(:\w+:|<[/\\]?3|[\(\)\\\D|\*\$][\-\^]?[:;=]|[:;=B8][\-\^]?[3DOPp@\$\*\\\)\(/|])(\s|[!\.\?]|$)"
        ).expect("To have valid regex");
    }
    clean_spaces(RE.replace_all(text, "").as_ref())
}

// fn remove_emoji(text: &str) -> String {
//    todo!();
// }


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_can_remove_duplicate_spaces() {
        assert_eq!("", clean_spaces(""));
        assert_eq!("", clean_spaces(" "));
        assert_eq!("", clean_spaces("      "));
        assert_eq!("hello", clean_spaces(" hello "));
        assert_eq!("hello world", clean_spaces(" hello world  "));
        assert_eq!("hello world", clean_spaces("hello  world"));
        assert_eq!("hello world, foo bar", clean_spaces("hello  world, foo    bar  "));
    }

    #[test]
    fn test_it_can_remove_user_mentions() {
        assert_eq!("", remove_user_mentions(""));
        assert_eq!("", remove_user_mentions(" "));
        assert_eq!("Hi there", remove_user_mentions("Hi there"));
        assert_eq!("Hi there", remove_user_mentions("Hi there @tester"));
        assert_eq!("Hi there test", remove_user_mentions("Hi there @tester test"));
        assert_eq!("Hi there test", remove_user_mentions("Hi there @tester test @this"));
    }

    #[test]
    fn test_it_can_remove_emoticons() {
        assert_eq!("", remove_emoticons(""));
        assert_eq!("", remove_emoticons(" "));
        assert_eq!("", remove_emoticons(":) "));
        assert_eq!("Hi", remove_emoticons("Hi :) "));
        assert_eq!("Hi", remove_emoticons("Hi :) "));
        assert_eq!("Hi", remove_emoticons("Hi :) :( :-( :D :p"));
        assert_eq!("Hi john", remove_emoticons("Hi :) john"));
    }

    #[test]
    fn test_it_can_detect_phatic_sentences() {
        let p = PhaticDetectorBuilder::new()
            .with_similarity_threshold(0.5)
            .build()
            .expect("To build detector instance");
        assert!(p.is_phatic("hello", &None).unwrap());
        assert!(!p.is_phatic("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.", &None).unwrap());
        assert!(p.is_phatic("thanks for the help sally", &None).unwrap());
        assert!(p.is_phatic("Hello, how can I help you?", &None).unwrap());
        assert!(!p.is_phatic("I can't login to my account.", &None).unwrap());
    }
}