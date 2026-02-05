#[derive(thiserror::Error, Debug)]
pub enum ModelError {
    #[error("build session: {0}")]
    OrtSessionBuildError(ort::Error),

    #[error("load session: {0}")]
    OrtSessionLoadError(ort::Error),

    #[error("load model: {0}")]
    OrtInputError(ort::Error),

    #[error("run inference: {0}")]
    OrtInferenceError(ort::Error),

    #[error("extract sensor: {0}")]
    OrtExtractSensorError(ort::Error),
}

#[derive(thiserror::Error, Debug)]
pub enum ImageError {
    #[error("load image: {0}")]
    OpenImageError(String),
}
