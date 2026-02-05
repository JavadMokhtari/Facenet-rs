mod configs;
mod detection;
mod models;
mod recognition;
mod utils;

use detection::YOLOFaceDetector;
use models::FaceBox;
use recognition::FaceNetModel;
use utils::read_image;

uniffi::setup_scaffolding!();

pub fn extract_embedding(image_path: &str) -> Vec<f32> {
    let mut detector = YOLOFaceDetector::new();
    let image = read_image(image_path);
    let output = detector.detect(&image);
    let face = output[0];

    let mut facenet = FaceNetModel::new();
    let embedding = facenet.extract_embedding(&image, face);
    embedding
}

#[uniffi::export]
pub fn detect_face(image_path: &str) -> Vec<FaceBox> {
    // let mut arrays = Vec::with_capacity(images.len());
    // for image in images {
    //     let result = detector.detect(&image);
    //         arrays.push(result);
    // }
    // arrays

    let mut detector = YOLOFaceDetector::new();
    let image = read_image(image_path);
    let output = detector.detect(&image);
    output
}
