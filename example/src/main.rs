use facenet::{detect_face, extract_embedding};

const IMAGE_PATH: &str = "../assets/cool_girl.jpg";

fn main() {
    let now = std::time::Instant::now();
    let result = extract_embedding(IMAGE_PATH);
    let elapsed = now.elapsed();
    println!("{:?}", result);
    println!("Elapsed: {:?}", elapsed);
}
