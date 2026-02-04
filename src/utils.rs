use std::path::Path;

use image::DynamicImage;

use crate::models::BoundingBox;

pub fn read_image(src: &str) -> DynamicImage {
    // Load image using standard image crate
    let img = image::open(Path::new(src)).unwrap();
    img
}

pub fn intersection(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    (box1.x2.min(box2.x2) - box1.x1.max(box2.x1)) * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
}

pub fn union(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    ((box1.x2 - box1.x1) * (box1.y2 - box1.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1))
        - intersection(box1, box2)
}
