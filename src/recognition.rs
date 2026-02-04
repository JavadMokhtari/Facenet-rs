use crate::configs;
use ort::session::Session;

pub fn extract_embeddings(images_dir: &str) {
    let mut session = Session::builder()
        .unwrap()
        .commit_from_file(configs::FACENET_MODEL_PATH)
        .unwrap();
}
