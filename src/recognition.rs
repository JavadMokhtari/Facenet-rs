use crate::configs::FacenetConfig;
use crate::models::FaceBox;
use crate::utils::preprocess_onnx_input;

use image::DynamicImage;
use ort::session::Session;
use ort::value::TensorRef;

pub struct FaceNetModel {
    pub session: Session,
    pub input_name: String,
    pub output_name: String,
}

impl FaceNetModel {
    pub fn init() -> Self {
        let session = Session::builder()
            .unwrap()
            .commit_from_file(FacenetConfig::MODEL_PATH)
            .unwrap();
        Self {
            session,
            input_name: "input_1".to_string(),
            output_name: "Bottleneck_BatchNorm".to_string(),
        }
    }

    pub fn extract_embedding(&mut self, image: &DynamicImage, face: FaceBox) -> Vec<f32> {
        let width = (face.x2 as u32).saturating_sub(face.x1 as u32);
        let height = (face.y2 as u32).saturating_sub(face.y1 as u32);

        let cropped = &image.crop_imm(face.x1 as u32, face.y1 as u32, width, height);
        cropped.save("cropped.png").unwrap();

        // let now = std::time::Instant::now();

        let input_tensor = preprocess_onnx_input(cropped, (160, 160));

        // let preproc_time = now.elapsed();
        // println!("Preprocessing time: {:?}", preproc_time);

        let input =
            ort::inputs![&self.input_name => TensorRef::from_array_view(&input_tensor).unwrap()];
        let outputs = self.session.run(input).unwrap();
        let output = outputs
            .get(&self.output_name.as_str())
            .unwrap()
            .try_extract_array::<f32>()
            .unwrap()
            .t()
            .into_owned();
        let embedding = output.into_raw_vec_and_offset().0;
        embedding
    }
}
