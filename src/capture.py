import time


class FrameGrabber:
    """Wraps cv2.VideoCapture with frame skipping.

    Every frame is returned for display, but only every Nth frame
    has should_process=True for running detection/tracking.
    """

    def __init__(self, cap, frame_stride=3):
        self._cap = cap
        self.frame_stride = frame_stride
        self._frame_idx = 0
        self.fps = cap.get(5) or 30.0  # CAP_PROP_FPS
        self.width = int(cap.get(3))   # CAP_PROP_FRAME_WIDTH
        self.height = int(cap.get(4))  # CAP_PROP_FRAME_HEIGHT

    def next(self):
        """Returns (frame, should_process, frame_idx) or (None, False, -1)."""
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None, False, -1
        idx = self._frame_idx
        should_process = (idx % self.frame_stride) == 0
        self._frame_idx += 1
        return frame, should_process, idx

    def raise_stride(self):
        """Increase stride for adaptive performance. Returns new stride."""
        steps = [3, 5, 8, 12]
        for s in steps:
            if s > self.frame_stride:
                self.frame_stride = s
                return s
        return self.frame_stride

    def release(self):
        self._cap.release()
