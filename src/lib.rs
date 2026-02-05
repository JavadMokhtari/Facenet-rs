mod configs;
mod detection;
mod models;
mod recognition;
mod utils;

use detection::YOLOFaceDetector;
use recognition::FaceNetModel;
use utils::read_image;

uniffi::setup_scaffolding!();

#[uniffi::export]
pub fn extract_embedding(image_path: String) -> Vec<f32> {
    let mut detector = YOLOFaceDetector::init();
    let image = read_image(&image_path);
    let output = detector.detect(&image);
    let face = output[0];
    println!("Detected face box {:?}", face);

    let mut facenet = FaceNetModel::init();
    let now = std::time::Instant::now();
    let embedding = facenet.extract_embedding(&image, face);
    println!("Feature extraction time taken: {:?}", now.elapsed());
    embedding
}

#[uniffi::export]
pub fn detect_face(image_path: String) -> Vec<Vec<f32>> {
    // let mut arrays = Vec::with_capacity(images.len());
    // for image in images {
    //     let result = detector.detect(&image);
    //         arrays.push(result);
    // }
    // arrays

    let mut detector = YOLOFaceDetector::init();
    let image = read_image(&image_path);
    let output = detector.detect(&image);
    let mut faces = Vec::new();
    for face in output {
        faces.push(vec![face.x1, face.y1, face.x2, face.y2, face.confidence]);
    }
    faces
}
