pub struct FaceDetectorConfig;

impl FaceDetectorConfig {
    pub const MODEL_PATH: &str = "../assets/yolov11s-face.onnx";
    pub const INPUT_SIZE: u32 = 320;
    pub const CONFIDENCE: f32 = 0.5;
    pub const IOU: f32 = 0.5;

}

pub struct FacenetConfig;

impl FacenetConfig {
    pub const MODEL_PATH: &str = "../assets/facenet128.onnx";
}