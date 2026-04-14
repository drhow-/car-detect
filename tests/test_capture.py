from src.capture import FrameGrabber


class FakeVideoCap:
    """Mock cv2.VideoCapture."""

    def __init__(self, frame_count=30):
        self._frame_count = frame_count
        self._idx = 0
        self._frame = __import__("numpy").zeros((720, 1280, 3), dtype=__import__("numpy").uint8)

    def isOpened(self):
        return self._idx < self._frame_count

    def read(self):
        if self._idx < self._frame_count:
            self._idx += 1
            return True, self._frame.copy()
        return False, None

    def get(self, prop_id):
        if prop_id == 5:  # CAP_PROP_FPS
            return 30.0
        if prop_id == 3:  # CAP_PROP_FRAME_WIDTH
            return 1280.0
        if prop_id == 4:  # CAP_PROP_FRAME_HEIGHT
            return 720.0
        return 0.0

    def release(self):
        pass


def test_frame_stride_skips_frames():
    cap = FakeVideoCap(frame_count=10)
    grabber = FrameGrabber(cap, frame_stride=3)
    results = []
    for _ in range(10):
        frame, should_process, idx = grabber.next()
        if frame is None:
            break
        results.append((should_process, idx))
    # Frames 0, 3, 6, 9 should be processed (every 3rd)
    process_indices = [idx for should, idx in results if should]
    assert process_indices == [0, 3, 6, 9]


def test_all_frames_returned_for_display():
    cap = FakeVideoCap(frame_count=6)
    grabber = FrameGrabber(cap, frame_stride=3)
    all_indices = []
    for _ in range(6):
        frame, _, idx = grabber.next()
        if frame is None:
            break
        all_indices.append(idx)
    assert all_indices == [0, 1, 2, 3, 4, 5]
