use facenet::run_feature_extractor;
use facenet::utils::read_image;
use facenet::YOLOFaceDetector;

const IMAGE_PATH: &str = "../assets/cool_girl.jpg";

fn main() {
    let img = read_image(IMAGE_PATH);
    let mut detector = YOLOFaceDetector::new();
    let output = detector.detect(&[img]);
    println!("{:?}", output);
}
