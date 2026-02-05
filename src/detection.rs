use std::path::Path;

use crate::configs::FaceDetectorConfig;
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use ndarray::{Array, Axis, Dim, Ix, s};
use ort::session::Session;
use ort::value::TensorRef;

use crate::models::FaceBox;
use crate::utils::{intersection, union};

pub struct YOLOFaceDetector {
    model: Session,
    input_name: String,
    output_name: String,
}

impl YOLOFaceDetector {
    pub fn new() -> YOLOFaceDetector {
        let session = Session::builder()
            .unwrap()
            .commit_from_file(FaceDetectorConfig::MODEL_PATH)
            .unwrap();
        Self {
            model: session,
            input_name: "images".to_string(),
            output_name: "output0".to_string(),
        }
    }

    pub fn from_config(model_path: &str) -> YOLOFaceDetector {
        let session = Session::builder()
            .unwrap()
            .commit_from_file(Path::new(model_path))
            .unwrap();
        Self {
            model: session,
            input_name: "images".to_string(),
            output_name: "output0".to_string(),
        }
    }

    fn preprocess(&self, image: &DynamicImage) -> Array<f32, Dim<[Ix; 4]>> {
        let resized = image.resize_exact(640, 640, FilterType::CatmullRom);
        let mut input = Array::zeros((1, 3, 640, 640));
        for pixel in resized.pixels() {
            let x = pixel.0 as _;
            let y = pixel.1 as _;
            let [r, g, b, _] = pixel.2.0;
            input[[0, 0, y, x]] = (r as f32) / 255.;
            input[[0, 1, y, x]] = (g as f32) / 255.;
            input[[0, 2, y, x]] = (b as f32) / 255.;
        }
        input
    }

    pub fn detect(&mut self, image: &DynamicImage) -> Vec<FaceBox> {
        let input_tensor = self.preprocess(image);
        // arrays.push(input.view());
        let input =
            ort::inputs![&self.input_name => TensorRef::from_array_view(&input_tensor).unwrap()];
        let outputs = self.model.run(input).unwrap();
        let output = outputs
            .get(&self.output_name.as_str())
            .unwrap()
            .try_extract_array::<f32>()
            .unwrap()
            .t()
            .into_owned();
        // }
        // let batch_tensor = ndarray::concatenate(Axis(0), &arrays).unwrap();
        //
        // // ndarray concatenate returns ArrayxD, we need Array4
        // let batch_tensor = batch_tensor.into_dimensionality::<ndarray::Ix4>().unwrap();
        //
        // // Ensure input is contiguous in memory (CowArray)
        // let input_contiguous = batch_tensor.as_standard_layout();
        //
        // // Create input tensor reference from ndarray view
        // let input_tensor = TensorRef::from_array_view(input_contiguous.view()).unwrap();
        //
        // // Run session - inputs! macro returns a Vec, not a Result
        // let inputs = ort::inputs![&self.input_name => input_tensor];
        //
        // let outputs = self.model.run(inputs).unwrap();
        //
        // let output = outputs
        //     .get(&self.output_name.as_str())
        //     .unwrap()
        //     .try_extract_array::<f32>()?
        //     .t()
        //     .into_owned();

        let mut boxes = Vec::new();
        let output = output.slice(s![.., .., 0]);
        let (img_width, img_height) = (image.width(), image.height());
        for row in output.axis_iter(Axis(0)) {
            let row: Vec<_> = row.iter().copied().collect();
            let (_, prob) = row
                .iter()
                // skip bounding box coordinates
                .skip(4)
                .enumerate()
                .map(|(index, value)| (index, *value))
                .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
                .unwrap();
            if prob < FaceDetectorConfig::CONFIDENCE {
                continue;
            }
            let xc = row[0] / 640. * (img_width as f32);
            let yc = row[1] / 640. * (img_height as f32);
            let w = row[2] / 640. * (img_width as f32);
            let h = row[3] / 640. * (img_height as f32);
            boxes.push(FaceBox {
                x1: xc - w / 2.,
                y1: yc - h / 2.,
                x2: xc + w / 2.,
                y2: yc + h / 2.,
                confidence: prob,
            });
        }

        boxes.sort_by(|box1, box2| box2.confidence.total_cmp(&box1.confidence));
        let mut result = Vec::new();

        while !boxes.is_empty() {
            result.push(boxes[0]);
            boxes = boxes
                .iter()
                .filter(|box1| {
                    intersection(&boxes[0], &box1) / union(&boxes[0], &box1)
                        < FaceDetectorConfig::IOU
                })
                .copied()
                .collect();
        }
        result
    }
}

//         let results = self.model.run(image).unwrap();
//         let mut faces = Vec::new();
//
//         for result in &results {
//             if let Some(ref boxes) = result.boxes {
//                 // We zip the rows of xyxy, xywh, and the confidence scores together
//                 let xyxy = boxes.xyxy();
//                 let confs = boxes.conf();
//
//                 for i in 0..confs.len() {
//                     faces.push(FaceBox {
//                         xyxy: [
//                             xyxy[i][0].max(0.0).round() as u16,
//                             xyxy[i][1].max(0.0).round() as u16,
//                             xyxy[i][2].max(0.0).round() as u16,
//                             xyxy[i][3].max(0.0).round() as u16,
//                         ],
//                         conf: confs[i],
//                     });
//                 }
//             }
//         }
//         faces
//     }
// }
