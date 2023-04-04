use regex::Regex;
use std::error::Error;

use rust_bert::pipelines::sentence_embeddings::builder::SentenceEmbeddingsBuilder;
use rust_bert::pipelines::sentence_embeddings::{Embedding, SentenceEmbeddingsModel, SentenceEmbeddingsModelType};

struct PhaticDetector {
    model: SentenceEmbeddingsModel,
    embeddings: Vec<Embedding>,
}

static EXAMPLES: &'static str = include_str!("phatic_examples.txt");

impl PhaticDetector {
    pub fn new() -> Result<PhaticDetector, Box<dyn Error>> {
        let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
            .create_model()?;

        let mut embeddings: Vec<Embedding> = vec![];
        for line in EXAMPLES.lines() {
            embeddings.extend(model.encode(&[line])?);
        }
        embeddings = normalize_all(embeddings);
        Ok(PhaticDetector { model, embeddings })
    }

    pub fn is_phatic(&self, text: &str) -> Result<bool, Box<dyn Error>> {
        match sanitise_text(text).split(char::is_whitespace).count() {
            0..=3 => Ok(true),
            15.. => Ok(false),
            _ => vector_check(&text, self)
        }
    }
}

fn normalize_all(embeddings: Vec<Embedding>) -> Vec<Embedding> {
    fn norm(a: &[f32]) -> Embedding {
        let z = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        a.iter().map(|x| x / z).collect()
    }
    embeddings.iter().map(|v| norm(v)).collect()
}

fn dot(a: &Embedding, b: &Embedding) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(a, b)| a * b).sum()
}

fn vector_check(text: &str, p: &PhaticDetector) -> Result<bool, Box<dyn Error>>
{
    let v = p.model.encode(&[text])?;
    let v = normalize_all(v);
    for e in &p.embeddings {
        if dot(&v[0], e) > 0.5 {
            return Ok(true);
        }
    }
    Ok(false)
}

fn sanitise_text(text: &str) -> String {
    // @todo is this correct? we're applying remove_user_mentions twice, and not removing other things...
    let text = remove_user_mentions(text);
    let text = remove_emoticons(&text);
    let text = remove_user_mentions(&text);
    text
}

fn clean_spaces(text: &str) -> String {
    let r = Regex::new(r" +").expect("To have valid regex"); // @todo should this be reused?
    r.replace_all(text, " ").trim().to_owned()
}

// fn remove_new_lines(text: &str) -> String {
//     todo!();
// }

fn remove_user_mentions(text: &str) -> String {
    let r = Regex::new(r"@[A-Za-z0-9]+").expect("To have valid regex"); // @todo should this be reused?
    clean_spaces(r.replace_all(text, "").as_ref())
}

fn remove_emoticons(text: &str) -> String {
    // @todo this has significant changes... better test suite?
    let r = Regex::new(r"(:\w+:|<[/\\]?3|[\(\)\\\D|\*\$][\-\^]?[:;=]|[:;=B8][\-\^]?[3DOPp@\$\*\\\)\(/|])(\s|[!\.\?]|$)")
        .expect("To have valid regex"); // @todo should this be reused?
    clean_spaces(r.replace_all(text, "").as_ref())
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
        let p = PhaticDetector::new().expect("To build detector instance");
        assert_eq!(true, p.is_phatic("hello").unwrap());
        assert_eq!(false, p.is_phatic("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.").unwrap());
        assert_eq!(true, p.is_phatic("thanks for the help sally").unwrap());
        assert_eq!(true, p.is_phatic("Hello, how can I help you?").unwrap());
        assert_eq!(false, p.is_phatic("I can't login to my account.").unwrap());
    }
}