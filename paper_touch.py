"""
四隅を検知し、指先を認識 & タッチできる
※失敗
"""
import cv2
import numpy as np

PW, PH = 800, 560
CHECKS = [(95, 170, 155, 230), (95, 280, 155, 340), (95, 390, 155, 450)]
LABELS = ["TASK 1", "TASK 2", "TASK 3"]
DWELL_FRAMES = 12


def ordered(pts):
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    return np.array([pts[np.argmin(s)], pts[np.argmax(d)],
                     pts[np.argmax(s)], pts[np.argmin(d)]], np.float32)


def detect_paper(frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 135)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 0.12 * w * h or area > 0.92 * w * h:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = ordered(approx)
        lengths = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
        if min(lengths) < 70:
            continue
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
        inner = cv2.erode(mask, np.ones((21, 21), np.uint8))
        brightness = cv2.mean(gray, mask=inner)[0]
        # A white sheet should be brighter than its surrounding desk.
        outer = cv2.dilate(mask, np.ones((31, 31), np.uint8))
        ring = cv2.subtract(outer, mask)
        surround = cv2.mean(gray, mask=ring)[0] if cv2.countNonZero(ring) else 0
        if brightness < 125 or brightness - surround < 15:
            continue
        candidates.append((area, quad))
    return max(candidates, key=lambda x: x[0])[1] if candidates else None


def fingertip_from_skin(paper_image):
    """Find a likely fingertip using skin segmentation and convex hull.
    Works best with a hand entering from the bottom of the camera image.
    """
    hsv = cv2.cvtColor(paper_image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(paper_image, cv2.COLOR_BGR2YCrCb)
    # Broad color ranges; lighting and skin tone affect reliability.
    mask_y = cv2.inRange(ycrcb, (0, 130, 75), (255, 180, 135))
    mask_h1 = cv2.inRange(hsv, (0, 20, 40), (25, 255, 255))
    mask_h2 = cv2.inRange(hsv, (165, 20, 40), (179, 255, 255))
    mask = cv2.bitwise_and(mask_y, cv2.bitwise_or(mask_h1, mask_h2))
    mask[:25, :] = 0
    mask[:, :20] = 0
    mask[:, -20:] = 0
    mask[-12:, :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 700]
    if not cnts:
        return None, mask
    # Choose a sizeable skin region. Uppermost hull point approximates fingertip
    # for a finger reaching in from the bottom edge of the paper.
    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt).reshape(-1, 2)
    if len(hull) == 0:
        return None, mask
    tip = tuple(map(int, hull[np.argmin(hull[:, 1])]))
    return tip, mask


def draw_panel():
    layer = np.zeros((PH, PW, 3), np.uint8)
    alpha = np.zeros((PH, PW), np.uint8)
    cv2.rectangle(layer, (40, 45), (760, 505), (246, 246, 246), -1)
    cv2.rectangle(alpha, (40, 45), (760, 505), 165, -1)
    cv2.putText(layer, "SMART PAPER - CHECKLIST", (75, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (45, 55, 65), 2, cv2.LINE_AA)
    for i, (x1, y1, x2, y2) in enumerate(CHECKS):
        cv2.rectangle(layer, (x1, y1), (x2, y2), (45, 95, 170), 4)
        cv2.putText(layer, LABELS[i], (200, y2 - 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.05, (40, 50, 65), 2, cv2.LINE_AA)
    return layer, alpha


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not found; try cv2.VideoCapture(1).")
    dest = np.float32([[0, 0], [PW - 1, 0], [PW - 1, PH - 1], [0, PH - 1]])
    locked_corners = None
    samples = []
    checked = [False] * 3
    dwell = [0] * 3
    previous_active = None
    ready_for_new_touch = True
    panel, alpha = draw_panel()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if locked_corners is None:
            corners = detect_paper(frame)
            if corners is not None:
                samples.append(corners)
                if len(samples) >= 12:
                    stack = np.stack(samples)
                    # Reject jittery detections before locking.
                    deviations = np.linalg.norm(stack - np.median(stack, axis=0), axis=2)
                    if np.percentile(deviations, 90) < 12:
                        locked_corners = np.median(stack, axis=0).astype(np.float32)
                    samples.clear()
            else:
                samples.clear()
            cv2.putText(frame, "Show full white paper on a dark desk",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (0, 160, 255), 2)
            if corners is not None:
                cv2.polylines(frame, [corners.astype(np.int32)], True,
                              (0, 200, 255), 2)
        if locked_corners is not None:
            H = cv2.getPerspectiveTransform(locked_corners, dest)
            H_inv = cv2.getPerspectiveTransform(dest, locked_corners)
            rectified = cv2.warpPerspective(frame, H, (PW, PH))
            tip, _ = fingertip_from_skin(rectified)

            active = None
            if tip is not None:
                tx, ty = tip
                for i, (x1, y1, x2, y2) in enumerate(CHECKS):
                    if x1 - 12 <= tx <= x2 + 12 and y1 - 12 <= ty <= y2 + 12:
                        active = i
                        break

            # Require an entry + dwell; leaving all boxes rearms the trigger.
            if active is None:
                dwell = [0] * 3
                ready_for_new_touch = True
            else:
                for i in range(3):
                    dwell[i] = dwell[i] + 1 if i == active else 0
                if ready_for_new_touch and dwell[active] >= DWELL_FRAMES:
                    checked[active] = not checked[active]
                    ready_for_new_touch = False
            previous_active = active

            layer = panel.copy()
            for i, (x1, y1, x2, y2) in enumerate(CHECKS):
                if checked[i]:
                    cv2.line(layer, (x1 + 10, y1 + 30), (x1 + 25, y1 + 45),
                             (35, 150, 70), 6, cv2.LINE_AA)
                    cv2.line(layer, (x1 + 25, y1 + 45), (x2 - 8, y1 + 12),
                             (35, 150, 70), 6, cv2.LINE_AA)
                if active == i:
                    cv2.rectangle(layer, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5),
                                  (0, 150, 240), 3)
            warped_layer = cv2.warpPerspective(layer, H_inv, (w, h))
            warped_alpha = cv2.warpPerspective(alpha, H_inv, (w, h))
            a = (warped_alpha.astype(np.float32) / 255.0)[:, :, None]
            frame = (frame * (1 - a) + warped_layer * a).astype(np.uint8)
            # Draw checkbox strokes opaquely in paper coordinates.
            strokes = np.zeros_like(layer)
            for i, (x1, y1, x2, y2) in enumerate(CHECKS):
                cv2.rectangle(strokes, (x1, y1), (x2, y2), (45, 95, 170), 4)
                if checked[i]:
                    cv2.line(strokes, (x1 + 10, y1 + 30), (x1 + 25, y1 + 45),
                             (35, 150, 70), 6, cv2.LINE_AA)
                    cv2.line(strokes, (x1 + 25, y1 + 45), (x2 - 8, y1 + 12),
                             (35, 150, 70), 6, cv2.LINE_AA)
                cv2.putText(strokes, LABELS[i], (200, y2 - 17),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.05, (40, 50, 65), 2, cv2.LINE_AA)
            warped_strokes = cv2.warpPerspective(strokes, H_inv, (w, h))
            stroke_mask = cv2.cvtColor(warped_strokes, cv2.COLOR_BGR2GRAY) > 5
            frame[stroke_mask] = warped_strokes[stroke_mask]
            if tip is not None:
                camera_tip = cv2.perspectiveTransform(
                    np.array([[tip]], np.float32), H_inv)[0, 0]
                cv2.circle(frame, tuple(camera_tip.astype(int)), 10,
                           (0, 255, 255), 3)
            cv2.putText(frame, "PAPER LOCKED | r: reset  c: clear",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (0, 180, 0), 2)
            cv2.polylines(frame, [locked_corners.astype(np.int32)], True,
                          (0, 180, 0), 2)
        cv2.imshow("Smart Paper Touch", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('r'):
            locked_corners = None
            samples.clear()
            dwell = [0] * 3
            ready_for_new_touch = True
        if key == ord('c'):
            checked = [False] * 3
            dwell = [0] * 3
            ready_for_new_touch = True
        if key == ord('s'):
            cv2.imwrite('smart_paper_touch_capture.png', frame)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
