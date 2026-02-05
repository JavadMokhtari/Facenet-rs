use std::path::Path;

use crate::models::FaceBox;
use image::DynamicImage;
use image::imageops::FilterType;
use ndarray::Array4;

pub fn read_image(src: &str) -> DynamicImage {
    // Load image using standard image crate
    let img = image::open(Path::new(src)).unwrap();
    img
}

pub fn preprocess_onnx_input(image: &DynamicImage, target_size: (u32, u32)) -> Array4<f32> {
    let (w, h) = target_size;

    let npx = (h * w) as usize;
    const SCALE: f32 = 1.0f32 / 255.0;

    // Resize and force RGB8
    let resized = image.resize_exact(w, h, FilterType::CatmullRom).to_rgb8();
    let raw = resized.into_raw(); // Vec<u8>, len == hw * 3

    // Precompute lut for u8 -> f32 scaled values
    let mut lut = [0f32; 256];
    for i in 0..256 {
        lut[i] = (i as f32) * SCALE;
    }

    // Single contiguous buffer in CHW ordering: [R..(hw), G..(hw), B..(hw)]
    let mut buf = vec![0f32; 3 * npx];

    for (i, px) in raw.chunks_exact(3).enumerate() {
        let r = lut[px[0] as usize];
        let g = lut[px[1] as usize];
        let b = lut[px[2] as usize];

        buf[i] = r; // channel 0
        buf[npx + i] = g; // channel 1
        buf[2 * npx + i] = b; // channel 2
    }

    // Convert once into ndarray NCHW (cheap)
    Array4::from_shape_vec((1, 3, h as usize, w as usize), buf).expect("shape matches")
}

pub fn intersection(box1: &FaceBox, box2: &FaceBox) -> f32 {
    (box1.x2.min(box2.x2) - box1.x1.max(box2.x1)) * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
}

pub fn union(box1: &FaceBox, box2: &FaceBox) -> f32 {
    ((box1.x2 - box1.x1) * (box1.y2 - box1.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1))
        - intersection(box1, box2)
}
