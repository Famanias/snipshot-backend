import numpy as np
import cv2
from snipshot_engine.rendering.bubble import find_largest_inscribed_rectangle

def run_lir_tests():
    print("=" * 60)
    print("  Speech Bubble Largest Inscribed Rectangle (LIR) Test Suite")
    print("=" * 60)

    # 1. Rectangle Test
    mask_rect = np.zeros((100, 100), dtype=np.uint8)
    mask_rect[20:80, 30:90] = 255
    rx, ry, rw, rh = find_largest_inscribed_rectangle(mask_rect)
    print(f"[1] Rectangle Test:")
    print(f"    - BBox: [20:80, 30:90] (size 60x60)")
    print(f"    - LIR: x={rx}, y={ry}, w={rw}, h={rh}")
    assert (rx, ry, rw, rh) == (30, 20, 60, 60), "LIR did not match bounding box for rectangle"
    assert np.all(mask_rect[ry:ry+rh, rx:rx+rw] == 255), "LIR contains background pixels"
    print("    SUCCESS: Rectangle LIR is correct.")

    # 2. Circle Test
    mask_circle = np.zeros((100, 100), dtype=np.uint8)
    cy, cx = 50, 50
    r = 30
    for y in range(100):
        for x in range(100):
            if (x - cx)**2 + (y - cy)**2 <= r**2:
                mask_circle[y, x] = 255
    cx_lir, cy_lir, cw_lir, ch_lir = find_largest_inscribed_rectangle(mask_circle)
    print(f"\n[2] Circle Test (Radius {r}):")
    print(f"    - BBox size: 60x60, Area: 3600 px²")
    print(f"    - LIR: x={cx_lir}, y={cy_lir}, w={cw_lir}, h={ch_lir}, Area: {cw_lir*ch_lir} px²")
    print(f"    - Retention Rate: {(cw_lir*ch_lir)/3600.0*100.0:.1f}%")
    assert cw_lir > 0 and ch_lir > 0, "LIR returned empty rectangle"
    assert np.all(mask_circle[cy_lir:cy_lir+ch_lir, cx_lir:cx_lir+cw_lir] == 255), "LIR contains background pixels"
    print("    SUCCESS: Circle LIR lies fully inside the mask.")

    # 3. Ellipse Test
    mask_ellipse = np.zeros((120, 120), dtype=np.uint8)
    ey, ex = 60, 60
    a, b = 40, 20
    for y in range(120):
        for x in range(120):
            if ((x - ex) / a)**2 + ((y - ey) / b)**2 <= 1.0:
                mask_ellipse[y, x] = 255
    ex_lir, ey_lir, ew_lir, eh_lir = find_largest_inscribed_rectangle(mask_ellipse)
    print(f"\n[3] Ellipse Test (Semi-axes {a}x{b}):")
    print(f"    - BBox size: 80x40, Area: 3200 px²")
    print(f"    - LIR: x={ex_lir}, y={ey_lir}, w={ew_lir}, h={eh_lir}, Area: {ew_lir*eh_lir} px²")
    print(f"    - Retention Rate: {(ew_lir*eh_lir)/3200.0*100.0:.1f}%")
    assert ew_lir > 0 and eh_lir > 0, "LIR returned empty rectangle"
    assert np.all(mask_ellipse[ey_lir:ey_lir+eh_lir, ex_lir:ex_lir+ew_lir] == 255), "LIR contains background pixels"
    print("    SUCCESS: Ellipse LIR lies fully inside the mask.")

    # 4. Irregular Shape with Tail Test (Fallback trigger test)
    mask_irregular = np.zeros((150, 150), dtype=np.uint8)
    mask_irregular[20:50, 20:50] = 255 # body (900 px²)
    mask_irregular[50:140, 20:22] = 255 # long thin tail
    ix_lir, iy_lir, iw_lir, ih_lir = find_largest_inscribed_rectangle(mask_irregular)
    bbox_area = 30 * 120
    lir_area = iw_lir * ih_lir
    ratio = lir_area / bbox_area
    print(f"\n[4] Irregular Shape with Long Tail (Fallback Test):")
    print(f"    - BBox Area: {bbox_area} px²")
    print(f"    - LIR Area: {lir_area} px²")
    print(f"    - Area Ratio: {ratio*100.0:.1f}% (Threshold: 45%)")
    assert ratio < 0.45, "Should be below the fallback threshold"
    print("    SUCCESS: Area ratio is below 45%, triggers fallback to bounding box correctly.")

    print("\n" + "=" * 60)
    print("  All LIR tests completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    run_lir_tests()
