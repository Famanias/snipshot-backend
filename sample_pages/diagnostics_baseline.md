# Baseline Diagnostics (Sample Pages)

Method: detection + OCR + merge + mask refinement + inpainting + rendering dispatch.
Translation mode: identity baseline (`translation = OCR text`) to isolate rendering behavior.

## Group Summary

| Complexity | Images | Regions | Bubble Detect % | Font Mean | Small Font <16% | Spacing px Mean | Overflow % | Shrink % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| complex | 4 | 64 | 25.0 | 30.8 | 14.1 | 4.5 | 100.0 | 100.0 |
| easy | 3 | 4 | 100.0 | 51.0 | 0.0 | 7.2 | 100.0 | 100.0 |
| medium | 3 | 23 | 34.8 | 36.1 | 0.0 | 4.9 | 100.0 | 100.0 |

## Per Image

| Image | Complexity | Regions | Bubble % | Font Mean | Small<16% | Overflow % | Shrink % |
|---|---|---:|---:|---:|---:|---:|---:|
| test-image-complex1.png | complex | 13 | 0.0 | 17.9 | 38.5 | 100.0 | 100.0 |
| test-image-complex2.jpg | complex | 10 | 10.0 | 50.6 | 0.0 | 100.0 | 100.0 |
| test-image-complex3.png | complex | 27 | 55.6 | 26.4 | 14.8 | 100.0 | 100.0 |
| test-image-complex4.png | complex | 14 | 0.0 | 37.0 | 0.0 | 100.0 | 100.0 |
| test-image-easy1.jpg | easy | 2 | 100.0 | 53.5 | 0.0 | 100.0 | 100.0 |
| test-image-easy2.jpg | easy | 1 | 100.0 | 49.0 | 0.0 | 100.0 | 100.0 |
| test-image-easy3.jpg | easy | 1 | 100.0 | 48.0 | 0.0 | 100.0 | 100.0 |
| test-image-medium.png | medium | 5 | 0.0 | 27.2 | 0.0 | 100.0 | 100.0 |
| test-image-medium1.png | medium | 8 | 100.0 | 50.0 | 0.0 | 100.0 | 100.0 |
| test-image-medium2.png | medium | 10 | 0.0 | 29.4 | 0.0 | 100.0 | 100.0 |

## Notes

- Overflow% counts regions where rendered temp text exceeds inner render box before centering/shrink.
- Shrink% counts regions where `_center_text_in_box` must downscale text (shrink factor < 1.0).
- Bubble Detect% is based on `detect_bubbles` returning a bubble rect for the region.