use facenet::extract_embedding;

fn main() {
    let image_path = "../assets/cool_girl.jpg".to_string();

    let now = std::time::Instant::now();
    let result = extract_embedding(image_path);
    let elapsed = now.elapsed();

    println!("Face embedding length: {:?}", result.len());
    println!("Process time: {:?}", elapsed);
}
