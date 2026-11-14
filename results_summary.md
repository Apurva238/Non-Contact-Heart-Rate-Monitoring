# Non-Contact Heart Rate Monitoring Using Facial Videos

## Project Overview

This project estimates heart rate from facial videos using Remote Photoplethysmography (rPPG) and digital signal processing techniques.

Instead of using a physical sensor, the system analyses subtle variations in facial skin colour caused by changes in blood volume during the cardiac cycle.

## Methodology

The implementation consists of the following stages:

1. **Video Input:** A facial video is provided as input.
2. **Face Detection and Tracking:** OpenCV Haar Cascade is used to detect and track the face.
3. **ROI Extraction:** Forehead, left cheek, and right cheek regions are identified.
4. **RGB Extraction:** Average red, green, and blue intensity values are calculated from facial skin regions across video frames.
5. **Preprocessing:** Forehead RGB signals are normalized to prepare them for further processing.
6. **PCA-Based rPPG Extraction:** Principal Component Analysis is applied to extract a representative signal. The first principal component (PC1) is selected.
7. **Heart Rate Estimation:** Welch's Power Spectral Density (PSD) method is used to identify the dominant frequency and convert it into beats per minute.

## Experimental Setup

| Parameter | Value |
|---|---|
| Video duration | 57 seconds |
| Frame rate | 30 FPS |
| Total frames | 1710 |
| Video resolution | 1920 × 1080 |
| Face detection | 1710/1710 frames |
| Valid forehead frames | 1710 |
| Selected PCA component | PC1 |

## Results

The complete pipeline successfully processed the input video.

**Estimated Heart Rate: 94.92 BPM**

The reference heart-rate data for the corresponding video interval had a mean of approximately 96.94 BPM.

The difference between the estimated value and the reference mean was approximately 2.02 BPM.

This is a preliminary result from a single video and does not establish general accuracy or clinical reliability.

## Limitations

- Performance may be affected by lighting variations and head movements.
- Facial skin colour variations can be very subtle.
- Haar Cascade detection may not perform reliably under all conditions.
- PCA does not guarantee that the selected component contains only pulse-related information.
- The current heart-rate estimator searches within a predefined frequency range of 85–110 BPM, limiting its applicability to heart rates outside that range.
- Evaluation on additional subjects and recordings is required.

## Future Scope

- Implement more advanced face detection and tracking methods.
- Explore POS and CHROM-based rPPG extraction.
- Improve motion and illumination artifact removal.
- Support a wider heart-rate range.
- Evaluate performance across multiple subjects and lighting conditions.
- Compare estimates against synchronized reference measurements.