mod configs;
mod detection;
mod models;
mod recognition;
pub mod utils;
mod errors;

use ort::ep::{CPUExecutionProvider, CUDAExecutionProvider};
use ort::session::Session;

use models::BoundingBox;
pub use detection::YOLOFaceDetector;

pub fn run_feature_extractor(){}



