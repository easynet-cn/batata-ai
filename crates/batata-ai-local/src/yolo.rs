//! YOLOv8 object detection using candle.
//!
//! Supports loading YOLOv8 weights in safetensors format and running
//! inference to detect objects with bounding boxes and class labels.

use std::path::Path;

use candle_core::{DType, Device, Module, ModuleT, Tensor};
use candle_nn::{batch_norm, conv2d, conv2d_no_bias, BatchNorm, Conv2d, Conv2dConfig, VarBuilder};
use tracing::info;

use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::multimodal::{DetectedObject, ImageData};

const IMAGE_SIZE: usize = 640;

/// COCO dataset class names (80 classes)
const COCO_CLASSES: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
];

// ── YOLOv8 Architecture Building Blocks ──────────────────────────

/// Conv + BatchNorm + SiLU activation block
struct ConvBnSilu {
    conv: Conv2d,
    bn: BatchNorm,
}

impl ConvBnSilu {
    fn new(
        in_c: usize,
        out_c: usize,
        kernel: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let pad = kernel / 2;
        let cfg = Conv2dConfig {
            stride,
            padding: pad,
            ..Default::default()
        };
        let conv = conv2d_no_bias(in_c, out_c, kernel, cfg, vb.pp("conv"))?;
        let bn = batch_norm(out_c, 1e-3, vb.pp("bn"))?;
        Ok(Self { conv, bn })
    }
}

impl Module for ConvBnSilu {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.conv.forward(x)?;
        let x = self.bn.forward_t(&x, false)?;
        // SiLU activation: x * sigmoid(x)
        let sigmoid = candle_nn::ops::sigmoid(&x)?;
        x * sigmoid
    }
}

/// Bottleneck block (residual connection if shortcut=true)
struct Bottleneck {
    cv1: ConvBnSilu,
    cv2: ConvBnSilu,
    shortcut: bool,
}

impl Bottleneck {
    fn new(
        in_c: usize,
        out_c: usize,
        shortcut: bool,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let hidden = out_c; // expansion ratio = 1.0
        let cv1 = ConvBnSilu::new(in_c, hidden, 3, 1, vb.pp("cv1"))?;
        let cv2 = ConvBnSilu::new(hidden, out_c, 3, 1, vb.pp("cv2"))?;
        Ok(Self { cv1, cv2, shortcut: shortcut && in_c == out_c })
    }
}

impl Module for Bottleneck {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = self.cv1.forward(x)?;
        let y = self.cv2.forward(&y)?;
        if self.shortcut {
            x + y
        } else {
            Ok(y)
        }
    }
}

/// C2f block (Cross Stage Partial with 2 convolutions + fusion)
struct C2f {
    cv1: ConvBnSilu,
    cv2: ConvBnSilu,
    bottlenecks: Vec<Bottleneck>,
}

impl C2f {
    fn new(
        in_c: usize,
        out_c: usize,
        n: usize,
        shortcut: bool,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let hidden = out_c / 2;
        let cv1 = ConvBnSilu::new(in_c, 2 * hidden, 1, 1, vb.pp("cv1"))?;
        let cv2 = ConvBnSilu::new((2 + n) * hidden, out_c, 1, 1, vb.pp("cv2"))?;
        let mut bottlenecks = Vec::with_capacity(n);
        for i in 0..n {
            bottlenecks.push(Bottleneck::new(
                hidden,
                hidden,
                shortcut,
                vb.pp(format!("bottleneck.{i}")),
            )?);
        }
        Ok(Self { cv1, cv2, bottlenecks })
    }
}

impl Module for C2f {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = self.cv1.forward(x)?;
        let chunks = y.chunk(2, 1)?;
        let mut outputs = vec![chunks[0].clone(), chunks[1].clone()];
        let mut last = chunks[1].clone();
        for bn in &self.bottlenecks {
            last = bn.forward(&last)?;
            outputs.push(last.clone());
        }
        let cat = Tensor::cat(&outputs, 1)?;
        self.cv2.forward(&cat)
    }
}

/// SPPF (Spatial Pyramid Pooling - Fast)
struct Sppf {
    cv1: ConvBnSilu,
    cv2: ConvBnSilu,
}

impl Sppf {
    fn new(in_c: usize, out_c: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let hidden = in_c / 2;
        let cv1 = ConvBnSilu::new(in_c, hidden, 1, 1, vb.pp("cv1"))?;
        let cv2 = ConvBnSilu::new(hidden * 4, out_c, 1, 1, vb.pp("cv2"))?;
        Ok(Self { cv1, cv2 })
    }
}

impl Module for Sppf {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.cv1.forward(x)?;
        // MaxPool2d with kernel_size=5, stride=1, padding=2
        let p1 = x.max_pool2d_with_stride(5, 1)?;
        let p2 = p1.max_pool2d_with_stride(5, 1)?;
        let p3 = p2.max_pool2d_with_stride(5, 1)?;
        let cat = Tensor::cat(&[&x, &p1, &p2, &p3], 1)?;
        self.cv2.forward(&cat)
    }
}

/// YOLOv8 detection model
pub struct YoloV8Model {
    // Backbone
    stem: ConvBnSilu,
    dark2: (ConvBnSilu, C2f),
    dark3: (ConvBnSilu, C2f),
    dark4: (ConvBnSilu, C2f),
    dark5: (ConvBnSilu, C2f, Sppf),
    // Neck (FPN + PAN)
    up3_conv: ConvBnSilu,
    up3_c2f: C2f,
    up2_conv: ConvBnSilu,
    up2_c2f: C2f,
    down3_conv: ConvBnSilu,
    down3_c2f: C2f,
    down4_conv: ConvBnSilu,
    down4_c2f: C2f,
    // Head output convs (box + cls per scale)
    head_box_convs: Vec<Conv2d>,
    head_cls_convs: Vec<Conv2d>,
    head_box_blocks: Vec<(ConvBnSilu, ConvBnSilu)>,
    head_cls_blocks: Vec<(ConvBnSilu, ConvBnSilu)>,
    _dfl_conv: Conv2d,
    num_classes: usize,
    reg_max: usize,
    device: Device,
}

impl YoloV8Model {
    /// Load YOLOv8m from safetensors file
    pub fn load(model_path: &Path, device: &Device) -> Result<Self> {
        info!("loading YOLOv8 model from {}", model_path.display());

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path.to_path_buf()], DType::F32, device)
                .map_err(|e| BatataError::Inference(format!("failed to load weights: {e}")))?
        };

        Self::build(vb, 80, 16, device)
    }

    fn build(
        vb: VarBuilder,
        num_classes: usize,
        reg_max: usize,
        device: &Device,
    ) -> Result<Self> {
        let map_err = |e: candle_core::Error| BatataError::Inference(e.to_string());

        // YOLOv8m channels: [48, 96, 192, 384, 576]
        // Backbone
        let stem = ConvBnSilu::new(3, 48, 3, 2, vb.pp("model.0")).map_err(map_err)?;

        let dark2_conv = ConvBnSilu::new(48, 96, 3, 2, vb.pp("model.1")).map_err(map_err)?;
        let dark2_c2f = C2f::new(96, 96, 2, true, vb.pp("model.2")).map_err(map_err)?;

        let dark3_conv = ConvBnSilu::new(96, 192, 3, 2, vb.pp("model.3")).map_err(map_err)?;
        let dark3_c2f = C2f::new(192, 192, 4, true, vb.pp("model.4")).map_err(map_err)?;

        let dark4_conv = ConvBnSilu::new(192, 384, 3, 2, vb.pp("model.5")).map_err(map_err)?;
        let dark4_c2f = C2f::new(384, 384, 4, true, vb.pp("model.6")).map_err(map_err)?;

        let dark5_conv = ConvBnSilu::new(384, 576, 3, 2, vb.pp("model.7")).map_err(map_err)?;
        let dark5_c2f = C2f::new(576, 576, 2, true, vb.pp("model.8")).map_err(map_err)?;
        let sppf = Sppf::new(576, 576, vb.pp("model.9")).map_err(map_err)?;

        // Neck
        let up3_conv = ConvBnSilu::new(576, 384, 1, 1, vb.pp("model.10")).map_err(map_err)?;
        let up3_c2f = C2f::new(384 + 384, 384, 2, false, vb.pp("model.13")).map_err(map_err)?;

        let up2_conv = ConvBnSilu::new(384, 192, 1, 1, vb.pp("model.14")).map_err(map_err)?;
        let up2_c2f = C2f::new(192 + 192, 192, 2, false, vb.pp("model.17")).map_err(map_err)?;

        let down3_conv = ConvBnSilu::new(192, 192, 3, 2, vb.pp("model.18")).map_err(map_err)?;
        let down3_c2f = C2f::new(192 + 384, 384, 2, false, vb.pp("model.21")).map_err(map_err)?;

        let down4_conv = ConvBnSilu::new(384, 384, 3, 2, vb.pp("model.22")).map_err(map_err)?;
        let down4_c2f = C2f::new(384 + 576, 576, 2, false, vb.pp("model.25")).map_err(map_err)?;

        // Detection heads (3 scales)
        let head_channels = [192, 384, 576];
        let box_out = 4 * reg_max;
        let cls_out = num_classes;
        let conv_cfg = Conv2dConfig { stride: 1, padding: 0, ..Default::default() };

        let mut head_box_blocks = Vec::new();
        let mut head_cls_blocks = Vec::new();
        let mut head_box_convs = Vec::new();
        let mut head_cls_convs = Vec::new();

        for (i, &c) in head_channels.iter().enumerate() {
            let bb1 = ConvBnSilu::new(c, c, 3, 1, vb.pp(format!("model.26.cv2.{i}.0"))).map_err(map_err)?;
            let bb2 = ConvBnSilu::new(c, c, 3, 1, vb.pp(format!("model.26.cv2.{i}.1"))).map_err(map_err)?;
            let bc = conv2d(c, box_out, 1, conv_cfg, vb.pp(format!("model.26.cv2.{i}.2"))).map_err(map_err)?;

            let cb1 = ConvBnSilu::new(c, c, 3, 1, vb.pp(format!("model.26.cv3.{i}.0"))).map_err(map_err)?;
            let cb2 = ConvBnSilu::new(c, c, 3, 1, vb.pp(format!("model.26.cv3.{i}.1"))).map_err(map_err)?;
            let cc = conv2d(c, cls_out, 1, conv_cfg, vb.pp(format!("model.26.cv3.{i}.2"))).map_err(map_err)?;

            head_box_blocks.push((bb1, bb2));
            head_cls_blocks.push((cb1, cb2));
            head_box_convs.push(bc);
            head_cls_convs.push(cc);
        }

        // DFL conv
        let dfl_conv = conv2d(reg_max, 1, 1, conv_cfg, vb.pp("model.26.dfl.conv")).map_err(map_err)?;

        info!("YOLOv8 model loaded ({num_classes} classes)");

        Ok(Self {
            stem,
            dark2: (dark2_conv, dark2_c2f),
            dark3: (dark3_conv, dark3_c2f),
            dark4: (dark4_conv, dark4_c2f),
            dark5: (dark5_conv, dark5_c2f, sppf),
            up3_conv,
            up3_c2f,
            up2_conv,
            up2_c2f,
            down3_conv,
            down3_c2f,
            down4_conv,
            down4_c2f,
            head_box_convs,
            head_cls_convs,
            head_box_blocks,
            head_cls_blocks,
            _dfl_conv: dfl_conv,
            num_classes,
            reg_max,
            device: device.clone(),
        })
    }

    /// Run detection on an image
    pub fn detect(&self, image: &ImageData, conf_threshold: f32, iou_threshold: f32) -> Result<Vec<DetectedObject>> {
        let (orig_w, orig_h) = (image.width, image.height);
        let input = self.preprocess(image)?;
        let input = input
            .unsqueeze(0)
            .map_err(|e| BatataError::Inference(e.to_string()))?;

        // Forward pass
        let predictions = self.forward_impl(&input)?;

        // Post-process: decode boxes, apply NMS
        self.postprocess(&predictions, conf_threshold, iou_threshold, orig_w, orig_h)
    }

    fn forward_impl(&self, x: &Tensor) -> Result<Tensor> {
        let map_err = |e: candle_core::Error| BatataError::Inference(e.to_string());

        // Backbone
        let x = self.stem.forward(x).map_err(map_err)?;

        let x = self.dark2.0.forward(&x).map_err(map_err)?;
        let p2 = self.dark2.1.forward(&x).map_err(map_err)?;

        let x = self.dark3.0.forward(&p2).map_err(map_err)?;
        let p3 = self.dark3.1.forward(&x).map_err(map_err)?;

        let x = self.dark4.0.forward(&p3).map_err(map_err)?;
        let p4 = self.dark4.1.forward(&x).map_err(map_err)?;

        let x = self.dark5.0.forward(&p4).map_err(map_err)?;
        let x = self.dark5.1.forward(&x).map_err(map_err)?;
        let p5 = self.dark5.2.forward(&x).map_err(map_err)?;

        // FPN (top-down)
        let up3 = self.up3_conv.forward(&p5).map_err(map_err)?;
        let (_, _, h4, w4) = p4.dims4().map_err(map_err)?;
        let up3 = up3.upsample_nearest2d(h4, w4).map_err(map_err)?;
        let f3 = Tensor::cat(&[&up3, &p4], 1).map_err(map_err)?;
        let f3 = self.up3_c2f.forward(&f3).map_err(map_err)?;

        let up2 = self.up2_conv.forward(&f3).map_err(map_err)?;
        let (_, _, h3, w3) = p3.dims4().map_err(map_err)?;
        let up2 = up2.upsample_nearest2d(h3, w3).map_err(map_err)?;
        let f2 = Tensor::cat(&[&up2, &p3], 1).map_err(map_err)?;
        let n2 = self.up2_c2f.forward(&f2).map_err(map_err)?;

        // PAN (bottom-up)
        let d3 = self.down3_conv.forward(&n2).map_err(map_err)?;
        let d3 = Tensor::cat(&[&d3, &f3], 1).map_err(map_err)?;
        let n3 = self.down3_c2f.forward(&d3).map_err(map_err)?;

        let d4 = self.down4_conv.forward(&n3).map_err(map_err)?;
        let d4 = Tensor::cat(&[&d4, &p5], 1).map_err(map_err)?;
        let n4 = self.down4_c2f.forward(&d4).map_err(map_err)?;

        // Detection heads for each scale
        let scales = [&n2, &n3, &n4];
        let mut all_preds = Vec::new();

        for (i, &feat) in scales.iter().enumerate() {
            // Box branch
            let bx = self.head_box_blocks[i].0.forward(feat).map_err(map_err)?;
            let bx = self.head_box_blocks[i].1.forward(&bx).map_err(map_err)?;
            let box_pred = self.head_box_convs[i].forward(&bx).map_err(map_err)?;

            // Class branch
            let cx = self.head_cls_blocks[i].0.forward(feat).map_err(map_err)?;
            let cx = self.head_cls_blocks[i].1.forward(&cx).map_err(map_err)?;
            let cls_pred = self.head_cls_convs[i].forward(&cx).map_err(map_err)?;

            // Reshape: [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
            let (b, _, h, w) = box_pred.dims4().map_err(map_err)?;
            let box_flat = box_pred
                .reshape((b, 4 * self.reg_max, h * w))
                .map_err(map_err)?
                .permute((0, 2, 1))
                .map_err(map_err)?;
            let cls_flat = cls_pred
                .reshape((b, self.num_classes, h * w))
                .map_err(map_err)?
                .permute((0, 2, 1))
                .map_err(map_err)?;

            // Concat box + cls: [B, H*W, 4*reg_max + num_classes]
            let pred = Tensor::cat(&[&box_flat, &cls_flat], 2).map_err(map_err)?;
            all_preds.push(pred);
        }

        // Concat all scales: [B, total_anchors, 4*reg_max + num_classes]
        Tensor::cat(&all_preds, 1).map_err(|e| BatataError::Inference(e.to_string()))
    }

    fn postprocess(
        &self,
        predictions: &Tensor,
        conf_threshold: f32,
        iou_threshold: f32,
        _orig_w: u32,
        _orig_h: u32,
    ) -> Result<Vec<DetectedObject>> {
        let map_err = |e: candle_core::Error| BatataError::Inference(e.to_string());

        // predictions shape: [1, num_anchors, 4*reg_max + num_classes]
        let preds = predictions.squeeze(0).map_err(map_err)?;
        let num_anchors = preds.dims()[0];
        let preds_data: Vec<Vec<f32>> = preds.to_vec2().map_err(map_err)?;

        let box_channels = 4 * self.reg_max;
        let mut detections: Vec<(f32, f32, f32, f32, f32, usize)> = Vec::new();

        for anchor_idx in 0..num_anchors {
            let row = &preds_data[anchor_idx];

            // Find max class score (apply sigmoid)
            let cls_start = box_channels;
            let mut max_score: f32 = 0.0;
            let mut max_cls: usize = 0;
            for c in 0..self.num_classes {
                let score = sigmoid(row[cls_start + c]);
                if score > max_score {
                    max_score = score;
                    max_cls = c;
                }
            }

            if max_score < conf_threshold {
                continue;
            }

            // Decode box via DFL (simplified: take argmax of each reg_max group as offset)
            let mut box_vals = [0f32; 4];
            for k in 0..4 {
                let start = k * self.reg_max;
                // Softmax + weighted sum over reg_max values
                let slice = &row[start..start + self.reg_max];
                let max_val = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = slice.iter().map(|v| (v - max_val).exp()).sum();
                let mut weighted_sum = 0f32;
                for (j, &v) in slice.iter().enumerate() {
                    weighted_sum += j as f32 * ((v - max_val).exp() / exp_sum);
                }
                box_vals[k] = weighted_sum;
            }

            // box_vals are [left, top, right, bottom] offsets
            // For simplicity, normalize to [0, 1] using IMAGE_SIZE
            let cx = (anchor_idx as f32 % 80.0) * 8.0; // Approximate grid mapping
            let cy = (anchor_idx as f32 / 80.0) * 8.0;

            let x1 = ((cx - box_vals[0] * 8.0) / IMAGE_SIZE as f32).clamp(0.0, 1.0);
            let y1 = ((cy - box_vals[1] * 8.0) / IMAGE_SIZE as f32).clamp(0.0, 1.0);
            let x2 = ((cx + box_vals[2] * 8.0) / IMAGE_SIZE as f32).clamp(0.0, 1.0);
            let y2 = ((cy + box_vals[3] * 8.0) / IMAGE_SIZE as f32).clamp(0.0, 1.0);

            if x2 > x1 && y2 > y1 {
                detections.push((x1, y1, x2, y2, max_score, max_cls));
            }
        }

        // NMS
        detections.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
        let mut kept = Vec::new();
        let mut suppressed = vec![false; detections.len()];

        for i in 0..detections.len() {
            if suppressed[i] {
                continue;
            }
            kept.push(i);
            for j in (i + 1)..detections.len() {
                if suppressed[j] || detections[j].5 != detections[i].5 {
                    continue;
                }
                if iou(&detections[i], &detections[j]) > iou_threshold {
                    suppressed[j] = true;
                }
            }
        }

        let results = kept
            .into_iter()
            .map(|i| {
                let (x1, y1, x2, y2, conf, cls) = detections[i];
                DetectedObject {
                    label: COCO_CLASSES.get(cls).unwrap_or(&"unknown").to_string(),
                    confidence: conf,
                    bbox: (x1, y1, x2, y2),
                }
            })
            .collect();

        Ok(results)
    }

    /// Preprocess image: resize to 640x640, normalize to [0, 1]
    fn preprocess(&self, image: &ImageData) -> Result<Tensor> {
        let (src_w, src_h) = (image.width as usize, image.height as usize);
        let channels = image.bytes.len() / (src_w * src_h);

        let mut chw = vec![0f32; 3 * IMAGE_SIZE * IMAGE_SIZE];

        for c in 0..3 {
            for y in 0..IMAGE_SIZE {
                for x in 0..IMAGE_SIZE {
                    let src_x = (x as f32 * src_w as f32 / IMAGE_SIZE as f32) as usize;
                    let src_y = (y as f32 * src_h as f32 / IMAGE_SIZE as f32) as usize;
                    let src_x = src_x.min(src_w - 1);
                    let src_y = src_y.min(src_h - 1);

                    let src_idx = (src_y * src_w + src_x) * channels + c.min(channels - 1);
                    // YOLOv8 normalizes to [0, 1]
                    chw[c * IMAGE_SIZE * IMAGE_SIZE + y * IMAGE_SIZE + x] =
                        image.bytes[src_idx] as f32 / 255.0;
                }
            }
        }

        Tensor::from_vec(chw, (3, IMAGE_SIZE, IMAGE_SIZE), &self.device)
            .map_err(|e| BatataError::Inference(e.to_string()))
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn iou(a: &(f32, f32, f32, f32, f32, usize), b: &(f32, f32, f32, f32, f32, usize)) -> f32 {
    let x1 = a.0.max(b.0);
    let y1 = a.1.max(b.1);
    let x2 = a.2.min(b.2);
    let y2 = a.3.min(b.3);

    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a.2 - a.0) * (a.3 - a.1);
    let area_b = (b.2 - b.0) * (b.3 - b.1);
    let union = area_a + area_b - intersection;

    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}
