"""
四隅認識 ＆ 画面投影
    s:認識画面のスクショが取れる
"""

import cv2
import numpy as np
from pathlib import Path

PAPER_W, PAPER_H = 900, 620


def order_points(points):
    """Return points in TL, TR, BR, BL order."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)

    top_left = pts[np.argmin(sums)]
    bottom_right = pts[np.argmax(sums)]
    top_right = pts[np.argmin(diffs)]
    bottom_left = pts[np.argmax(diffs)]

    return np.array(
        [top_left, top_right, bottom_right, bottom_left],
        dtype=np.float32
    )


def find_paper_corners(frame):
    """
    Find the largest 4-corner contour that looks like a paper sheet.
    Returns ordered corners or None.
    """
    image_area = frame.shape[0] * frame.shape[1]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:30]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.08:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            corners = order_points(approx.reshape(4, 2))

            # Reject degenerate quadrilaterals.
            width1 = np.linalg.norm(corners[1] - corners[0])
            width2 = np.linalg.norm(corners[2] - corners[3])
            height1 = np.linalg.norm(corners[3] - corners[0])
            height2 = np.linalg.norm(corners[2] - corners[1])
            if min(width1, width2, height1, height2) < 40:
                continue

            return corners

    return None


def make_overlay(w, h):
    layer = np.full((h, w, 3), 255, dtype=np.uint8)

    # Header
    cv2.rectangle(layer, (0, 0), (w, 92), (35, 45, 60), -1)
    cv2.putText(layer, "SMART PAPER", (35, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(layer, "Markerless paper tracking demo", (35, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 220, 230), 1,
                cv2.LINE_AA)

    # Checklist
    cv2.rectangle(layer, (35, 125), (w - 35, 300), (240, 247, 255), -1)
    cv2.rectangle(layer, (35, 125), (w - 35, 300), (80, 135, 190), 2)
    cv2.putText(layer, "PROJECT CHECKLIST", (60, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 75, 115), 2,
                cv2.LINE_AA)

    tasks = [
        ("Collect reference images", True),
        ("Detect paper corners", True),
        ("Overlay digital content", False),
    ]
    y = 210
    for label, done in tasks:
        color = (50, 160, 90) if done else (80, 135, 190)
        cv2.rectangle(layer, (62, y - 18), (88, y + 8), color, 2)
        if done:
            cv2.line(layer, (67, y - 5), (75, y + 2), color, 2)
            cv2.line(layer, (75, y + 2), (84, y - 12), color, 2)
        cv2.putText(layer, label, (105, y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (45, 50, 60), 1,
                    cv2.LINE_AA)
        y += 38

    # Example graph
    cv2.rectangle(layer, (35, 330), (w - 35, h - 35), (250, 250, 250), -1)
    cv2.rectangle(layer, (35, 330), (w - 35, h - 35), (180, 185, 190), 1)
    cv2.putText(layer, "SAMPLE DATA", (60, 370),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 60, 70), 2,
                cv2.LINE_AA)

    left, top = 75, 405
    right, bottom = w - 75, h - 70
    cv2.line(layer, (left, bottom), (right, bottom), (100, 100, 100), 1)
    cv2.line(layer, (left, top), (left, bottom), (100, 100, 100), 1)

    values = [0.20, 0.34, 0.28, 0.55, 0.48, 0.72, 0.62, 0.86, 0.78]
    points = []
    for i, value in enumerate(values):
        x = left + int(i * (right - left) / (len(values) - 1))
        y = bottom - int(value * (bottom - top))
        points.append((x, y))

    for p1, p2 in zip(points, points[1:]):
        cv2.line(layer, p1, p2, (60, 145, 210), 4, cv2.LINE_AA)
    for point in points:
        cv2.circle(layer, point, 5, (35, 95, 160), -1, cv2.LINE_AA)

    cv2.putText(layer, "Move or tilt the paper to test tracking.",
                (60, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (90, 95, 100), 1, cv2.LINE_AA)
    return layer


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Check camera permissions or try VideoCapture(1)."
        )

    source_corners = np.float32([
        [0, 0],
        [PAPER_W - 1, 0],
        [PAPER_W - 1, PAPER_H - 1],
        [0, PAPER_H - 1],
    ])
    overlay = make_overlay(PAPER_W, PAPER_H)

    print("Show a plain sheet of paper against a contrasting desk.")
    print("Keep all four edges visible. Press q/ESC to quit, s to save.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        corners = find_paper_corners(frame)

        if corners is not None:
            # Draw detected outline for visual debugging.
            cv2.polylines(frame, [corners.astype(np.int32)], True,
                          (0, 220, 0), 3, cv2.LINE_AA)

            # Map the virtual paper rectangle to the detected paper quadrilateral.
            H = cv2.getPerspectiveTransform(source_corners, corners)
            warped = cv2.warpPerspective(
                overlay, H, (frame.shape[1], frame.shape[0])
            )

            # Blend overlay lightly so the real paper remains visible.
            # Only pixels that differ from the white background are overlaid.
            gray_overlay = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            mask = gray_overlay < 248
            alpha = np.clip((255 - gray_overlay).astype(np.float32) / 255.0,
                            0.0, 0.88)
            alpha3 = alpha[..., None]
            blended = (
                frame.astype(np.float32) * (1 - alpha3)
                + warped.astype(np.float32) * alpha3
            ).clip(0, 255).astype(np.uint8)
            frame[mask] = blended[mask]

            cv2.putText(frame, "PAPER DETECTED", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 0), 2,
                        cv2.LINE_AA)
        else:
            cv2.putText(frame, "Paper not detected - show all 4 edges",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 140, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Use a plain sheet on a darker, uncluttered desk",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (0, 140, 255), 1, cv2.LINE_AA)

        cv2.imshow("Smart Paper Demo - Markerless", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            filename = "smart_paper_markerless_screenshot.png"
            cv2.imwrite(filename, frame)
            print(f"Saved screenshot: {Path(filename).resolve()}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
