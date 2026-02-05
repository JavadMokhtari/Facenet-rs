use crate::configs::FacenetConfig;
use crate::models::FaceBox;
use image::DynamicImage;
use image::imageops::FilterType;
use ndarray::Array4;
use ort::session::Session;
use ort::value::TensorRef;

pub struct FaceNetModel {
    session: Session,
    input_name: String,
    output_name: String,
}

impl FaceNetModel {
    pub fn new() -> Self {
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
    fn preprocess(&self, image: &DynamicImage) -> Array4<f32> {
        const H: usize = 160;
        const W: usize = 160;
        const HW: usize = H * W;
        const SCALE: f32 = 1.0f32 / 255.0;

        // Resize and force RGB8
        let resized = image
            .resize_exact(W as u32, H as u32, FilterType::CatmullRom)
            .to_rgb8();
        let raw = resized.into_raw(); // Vec<u8>, len == hw * 3

        // Precompute lut for u8 -> f32 scaled values
        let mut lut = [0f32; 256];
        for i in 0..256 {
            lut[i] = (i as f32) * SCALE;
        }

        // Single contiguous buffer in CHW ordering: [R..(hw), G..(hw), B..(hw)]
        let mut buf = vec![0f32; 3 * HW];

        for (i, px) in raw.chunks_exact(3).enumerate() {
            let r = lut[px[0] as usize];
            let g = lut[px[1] as usize];
            let b = lut[px[2] as usize];

            buf[i] = r; // channel 0
            buf[HW + i] = g; // channel 1
            buf[2 * HW + i] = b; // channel 2
        }

        // Convert once into ndarray (cheap)
        Array4::from_shape_vec((1, 3, H, W), buf).expect("shape matches")
    }

    pub fn extract_embedding(&mut self, image: &DynamicImage, face: FaceBox) -> Vec<f32> {
        let width = (face.x2 as u32).saturating_sub(face.x1 as u32);
        let height = (face.y2 as u32).saturating_sub(face.y1 as u32);

        let cropped = &image.crop_imm(face.x1 as u32, face.y1 as u32, width, height);
        cropped.save("cropped.png").unwrap();

        // let now = std::time::Instant::now();
        let input_tensor = self.preprocess(cropped);
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
