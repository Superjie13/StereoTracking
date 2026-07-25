## Reproducibility

This section summarizes the dataset splits, preprocessing parameters, model hyperparameters, training and inference configurations, tracker settings, and software environment used in the paper.

### 1. Dataset and Evaluation Protocols

| Item | AirDrone | KITTI Tracking | Drone-vs-Bird |
|---|---|---|---|
| Dataset type | Synthetic stereo airborne MOT dataset | Real-world stereo road-scene MOT dataset | Real-world monocular airborne-object dataset |
| Number of sequences | 64 | 21 annotated training sequences | 20 selected single-UAV sequences from the original 77 sequences |
| Total stereo/image pairs | 16,821 stereo pairs | 8,008 stereo pairs | Sequence-dependent |
| Training split | 19 complete sequences, 5,846 stereo pairs | First temporal half of each annotated sequence | First temporal half of each selected sequence |
| Evaluation split | 45 complete sequences, 10,975 stereo pairs | Second temporal half of each annotated sequence | Second temporal half of each selected sequence |
| Split principle | Fixed sequence-level split; no sequence or frame overlap between training and evaluation | Temporal split following the adopted benchmark protocol | Temporal split of each selected single-UAV sequence |
| Input resolution | \(1280 \times 720\) | \(1600 \times 576\) | \(1280 \times 720\) |
| Original acquisition rate | 30 Hz | Dataset default | Dataset default |
| MOT evaluation rate | 10 fps | 10 fps | 10 fps |
| Depth source | StereoSGBM disparity and simulator ground-truth depth | Stereo images | monocularly estimated depth using DepthAnythingV2 |
| AirDrone camera baseline | 0.25 m | Dataset calibration | Not applicable |
| AirDrone camera field of view | \(90^\circ\) | Dataset calibration | Dataset default |
| AirDrone evaluation depth range | 0–80 m | Dataset-specific evaluation | Dataset-specific evaluation |
| Annotation format | MOTChallenge-style and COCO-style annotations | Converted to the adopted MOT format | Converted to the adopted MOT format |
| Evaluation metrics | HOTA, CLEAR MOT, Identity metrics, and COCO detection metrics | HOTA, CLEAR MOT, Identity metrics, and COCO detection metrics | HOTA, CLEAR MOT, Identity metrics, and COCO detection metrics |

The AirDrone split is performed at the sequence level rather than by dividing
individual sequences into temporal segments. This prevents temporal and
sequence-level leakage between training and evaluation.

<details>
<summary><b>AirDrone training sequences: 19 sequences</b></summary>

```text
0000
0001
2023-03-02-18-56-46
2023-03-02-19-11-46
2023-03-02-19-24-22
2023-03-02-19-30-16
2023-03-02-20-00-03
2023-03-02-20-03-07
2023-03-02-22-44-04
2023-03-02-22-50-34
2023-03-09-16-11-04
2023-03-09-16-48-10
2023-03-09-16-53-25
2023-03-09-16-59-12
2023-03-09-17-05-47
2023-03-09-17-11-11
2023-03-09-17-16-28
2023-03-23-16-32-10
2023-03-23-17-08-02
```

</details>

<details>
<summary><b>AirDrone evaluation sequences: 45 sequences</b></summary>

```text
2023-03-02-18-16-23
2023-03-02-19-15-09
2023-03-02-19-27-33
2023-03-02-19-33-05
2023-03-02-19-36-01
2023-03-02-22-38-01
2023-03-20-15-32-49
2023-03-20-15-40-43
2023-03-20-15-59-35
2023-03-20-16-07-43
2023-03-20-16-17-32
2023-03-23-16-53-48
2023-03-23-16-54-34
2023-03-23-17-01-04
2024-06-04-12-00-17
2024-06-04-12-11-41
2024-06-04-14-57-56
2024-06-04-15-03-19
2024-06-04-15-08-45
2024-06-04-15-14-20
2024-06-04-15-22-04
2024-06-04-15-27-10
2024-06-04-15-32-24
2024-06-04-15-37-55
2024-06-04-15-43-39
2024-06-04-15-48-48
2024-06-04-15-54-17
2024-06-04-15-59-58
2024-06-04-16-05-02
2024-06-04-16-10-11
2024-06-04-16-15-28
2024-06-04-16-20-54
2024-06-04-16-26-33
2024-06-04-16-32-08
2024-06-04-16-37-47
2024-06-04-16-43-07
2024-06-04-16-48-20
2024-06-04-16-53-36
2024-06-04-16-58-49
2024-06-04-17-04-02
2024-06-04-17-09-31
2024-06-04-17-14-56
2024-06-04-17-20-10
2024-06-04-17-25-38
2024-06-04-17-30-43
```

</details>

<details>
<summary><b>Selected Drone-vs-Bird sequences: 20 sequences</b></summary>

```text
dji_mavick_close_buildings
dji_phantom_mountain_cross
gopro_004
gopro_005
GOPR5842_002
GOPR5842_005
GOPR5842_007
GOPR5844_002
GOPR5844_004
GOPR5845_001
GOPR5845_004
GOPR5846_002
GOPR5846_005
GOPR5847_003
GOPR5847_004
GOPR5848_002
GOPR5848_004
dji_matrice_210_hillside
dji_matrice_210_mountain
dji_mavick_mountain
```

Each selected Drone-vs-Bird sequence is divided temporally into two equal
portions. The first half is used for training and the second half for
evaluation.

</details>

---

### 2. Randomness and Runtime Determinism

| Parameter | Setting |
|---|---|
| Global random seed | 0 |
| NumPy random seed | 0 |
| PyTorch random seed | 0 |
| Dataset sampler seed | 0 |
| Data-shuffling seed | 0 |
| Data-augmentation seed | 0 |
| `cudnn_benchmark` | `False` |
| Deterministic mode | `False` |
| Distributed training | Disabled |
| Number of GPUs | 1 |

A fixed seed of 0 is used to control model initialization, data shuffling,
sampling, and stochastic augmentation. Since deterministic mode is disabled,
exact bitwise-identical results across different GPUs, CUDA versions, or
operating systems are not guaranteed. The fixed seed nevertheless improves
run-to-run consistency under the same environment.

---

### 3. StereoSGBM Configuration

AirDrone disparity maps are generated using OpenCV StereoSGBM with the
following fixed configuration.

| Parameter | Value |
|---|---:|
| OpenCV function | `cv2.StereoSGBM_create` |
| `minDisparity` | 0 |
| `numDisparities` | 48 |
| `blockSize` | 3 |
| `P1` | 96 |
| `P2` | 384 |
| `disp12MaxDiff` | 0 |
| `uniquenessRatio` | 10 |
| `speckleWindowSize` | 400 |
| `speckleRange` | 10 |
| `preFilterCap` | 63 |
| `mode` | `cv2.STEREO_SGBM_MODE_SGBM_3WAY` |
| Invalid disparity value | 0 |
| OpenCV fixed-point conversion | Disparity output divided by 16 |
| Metric-depth conversion | \(d_i = B F / (\mathrm{disp}_i + \epsilon)\) |


To reduce preprocessing ambiguity and computational burden, the AirDrone
dataset includes the **precomputed disparity maps used in all reported
experiments**. Therefore, users can reproduce the reported detection and
tracking results without rerunning stereo matching.

---

### 4 Camera-Motion-Aware Correction

| Parameter | Setting |
|---|---|
| Optical-flow method | Farneback dense optical flow |
| Input frames | Consecutive left-camera RGB frames |
| Optical-flow input resolution | \(256 \times 256\) |
| `pyr_scale` | 0.5 |
| `levels` | 5 |
| `winsize` | 128 |
| `iterations` | 3 |
| `poly_n` | 5 |
| `poly_sigma` | 1.1 |
| Directional-consistency threshold \(\tau_1\) | 60% |
| Minimum valid-flow ratio \(\tau_2\) | 60% |
| Mesh-flow aggregation | Median flow within each grid cell |
| Background-motion estimation | Mean of valid mesh flows |
| Compensation stage | Applied during association of unmatched tracks and detections |

Dense flow is first aggregated into mesh flows using median filtering.
Mesh flows inconsistent with the dominant motion direction are rejected.
Camera-motion compensation is applied only when the valid mesh-flow ratio
exceeds \(\tau_2\).


### 5. Multi-Modal Detector

| Item | Setting |
|---|---|
| Detector family | YOLOX-S |
| Detection classes | 1 (`drone`) |
| Input modalities | RGB image and stereo-derived disparity/depth representation |
| Backbone depth factor | 0.33 |
| Backbone width factor | 0.5 |
| Feature channels | 256 |
| Feature-map strides | 8, 16, and 32 |
| Stacked head convolutions | 2 |
| Activation | SiLU |
| Normalization | Batch normalization |
| Pretrained initialization | COCO-pretrained YOLOX-S |
| Pretrained checkpoint | `yolox_s_8xb8-300e_coco_20220917_030738-d7e60cb2.pth` |

The disparity/depth input is converted to a three-channel representation before
being passed to the depth branch.

---

### 6. Detector Training Configuration

| Category | Parameter | Setting |
|---|---|---|
| Input | Training resolution | \(1280 \times 720\) |
| Input | Padding divisor | 32 |
| Input | RGB padding value | 114 |
| Training | Number of epochs | 50 |
| Training | Batch size per GPU | 8 |
| Training | Number of GPUs | 1 |
| Training | Training workers | 16 |
| Training | Validation workers | 2 |
| Optimizer | Type | SGD |
| Optimizer | Initial learning rate | \(1\times10^{-3}\) |
| Optimizer | Momentum | 0.9 |
| Optimizer | Nesterov momentum | Enabled |
| Optimizer | Weight decay | \(5\times10^{-4}\) |
| Optimizer | Normalization-layer weight decay | 0 |
| Optimizer | Bias weight decay | 0 |
| LR schedule | Warm-up | Quadratic warm-up during epochs 0–2 |
| LR schedule | Main schedule | Cosine annealing during epochs 2–45 |
| LR schedule | Minimum learning rate | \(5\times10^{-5}\) |
| LR schedule | Final stage | Constant learning rate during epochs 45–50 |
| EMA | Enabled | Yes |
| EMA | Momentum | \(1\times10^{-4}\) |
| EMA | Update buffers | `True` |
| YOLOX schedule | Final augmentation-free stage | Last 5 epochs |
| Assignment | Training assigner | SimOTA |
| Assignment | Center radius | 2.5 |
| Bounding-box loss | IoU loss weight | 5.0 |
| Classification loss | Sigmoid cross-entropy weight | 1.0 |
| Objectness loss | Sigmoid cross-entropy weight | 1.0 |
| Auxiliary loss | L1 loss weight | 1.0 |
| Annotation filtering | Minimum bounding-box width/height | \(1 \times 1\) pixel |

#### RGB–Depth-Consistent Data Augmentation

To support joint RGB and depth inputs, the standard MixUp operation is
redesigned as **RGB–depth-consistent MixUp**.

| Augmentation | Setting |
|---|---|
| RGB–depth-consistent MixUp | Enabled during the first 45 epochs |
| MixUp scale-ratio range | 0.8–1.6 |
| Cross-modal synchronization | Identical sampling coefficients and spatial transformations are applied to RGB, disparity/depth, bounding boxes, and labels |
| Purpose | Preserve pixel-level RGB–depth alignment during augmentation |
| HSV augmentation | YOLOX HSV random augmentation applied to RGB images |
| Horizontal flipping | Probability 0.5 |
| Flip synchronization | The same flip is applied to RGB, disparity/depth, and annotations |
| Final five epochs | MixUp disabled; synchronized HSV augmentation and horizontal flipping retained |

---

### 7. Detector Inference Configuration

| Parameter | Setting |
|---|---|
| Input resolution | \(1280 \times 720\) |
| Inference batch size | 1 |
| Inference workers | 2 |
| Detection score threshold | 0.01 |
| NMS method | Standard NMS |
| NMS IoU threshold | 0.5 |
| Maximum detections per image | 300 |
| Test-time augmentation | Disabled |
| AirDrone evaluation depth range | 0–80 m |

The detector threshold of 0.01 is the pre-tracking detector output threshold.
Each tracker subsequently applies its own object-confidence and track-
initialization thresholds.

---

### 8. OCSORT Tracking Configuration

The following settings correspond to the OCSORT configuration used for the
reported AirDrone experiments.

| Parameter | Setting |
|---|---:|
| Tracker | OCSORT |
| Motion model | Kalman filter |
| `obj_score_thr` | 0.3 |
| `init_track_thr` | 0.7 |
| `match_iou_thr` | 0.1 |
| `num_frames_retain` | 30 |
| `num_tentatives` | 3 |
| `vel_consist_weight` | 0.2 |
| `vel_delta_t` | 3 |
| `weight_iou_with_det_scores` | `False` |

The parameter names follow the MMTracking conventions. The executable tracker configurations are located under:

```text
configs/stereo_tracking/
├── ocsort/
└── ...
```

---

### 9. Evaluation Configuration

| Evaluation target | Implementation/metrics |
|---|---|
| Object detection | COCO-style bounding-box evaluation |
| Detection metrics | mAP, mAP50, and mAP75 |
| Multi-object tracking | TrackEval/MMTracking-compatible evaluation |
| Tracking metrics | HOTA, DetA, AssA, MOTA, IDF1, ID switches, and related CLEAR/Identity metrics |
| AirDrone distance filtering | Only targets within 0–80 m are evaluated |
| Frame rate | All MOT experiments standardized to 10 fps |
| Evaluation batch size | 1 |

---

### 10. Software Environment

| Component | Version/setting |
|---|---|
| Reference operating system | Ubuntu 20.04 / Linux |
| Python | 3.9; experiment log: 3.9.23 |
| pip | 22.2.2 in the reference installation |
| PyTorch | 1.13.1 |
| TorchVision | 0.14.1 |
| TorchAudio | 0.13.1 |
| CUDA runtime | 11.7 |
| cuDNN | 8.5 |
| OpenCV | 4.10.0 |
| MMTracking | 1.0.0rc1, customized in this repository |
| MMEngine | 0.10.3 |
| MMClassification | 1.0.0rc4 |
| MMCV | 2.0.0rc3 |
| MMDetection | 3.0.0rc4 |
| MMYOLO | 0.2.0 |
| Evaluation package | TrackEval |
| GPU | NVIDIA GeForce RTX 4090, 24 GB |
| Number of GPUs | 1 |
| cuDNN benchmark | Disabled |

---