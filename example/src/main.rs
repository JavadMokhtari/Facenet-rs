use facenet::{detect_face_from_dir, extract_embedding};
use std::time::Instant;

fn main() {
    // let image = "../assets/cool_girl.jpg";
    // let embedding = extract_embedding(image);
    // println!("Length of embedding vector: {:?}", embedding.len());

    let image_dir = "C:/Users/j.mokhtari/Pictures/faces";
    let now = Instant::now();
    let results = detect_face_from_dir(image_dir);
    println!("Processed {} images in {:?}", results.len(), now.elapsed());
    for result in results {
        println!("{:?}", result);
    }
}
