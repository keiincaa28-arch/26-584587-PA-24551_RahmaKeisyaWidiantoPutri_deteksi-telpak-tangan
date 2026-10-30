
import cv2
import asyncio
import time

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
from ultralytics import YOLO

# Load hand detection model
model = YOLO("hand_yolov8n.pt")

# Store active connections
pcs = set()


class WebcamStream(VideoStreamTrack):

    def __init__(self):
        super().__init__()

        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        self.frame_count = 0
        self.last_boxes = []

        if not self.camera.isOpened():
            raise RuntimeError("Webcam cannot be opened")

    async def recv(self):

        pts, time_base = await self.next_timestamp()

        # Read webcam frame
        ret, frame = await asyncio.to_thread(
            self.camera.read
        )

        if not ret:
            raise RuntimeError("Cannot read webcam")

        self.frame_count += 1

        # Run YOLO every 2 frames to reduce computation
        if self.frame_count % 2 == 0:

            results = await asyncio.to_thread(
                model.predict,
                frame,
                conf=0.35,
                imgsz=640,
                verbose=False
            )

            self.last_boxes = []

            for box in results[0].boxes:

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                label = model.names[class_id]

                self.last_boxes.append(
                    (x1, y1, x2, y2, label, confidence)
                )

        # Draw bounding boxes
        for x1, y1, x2, y2, label, confidence in self.last_boxes:

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            text = f"{label}: {confidence:.2f}"

            cv2.putText(
                frame,
                text,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Convert image to WebRTC video frame
        video_frame = VideoFrame.from_ndarray(
            frame,
            format="bgr24"
        )

        video_frame.pts = pts
        video_frame.time_base = time_base

        return video_frame

    def stop(self):
        self.camera.release()
        super().stop()


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>UAV Video Streaming</title>
</head>
<body>
    <h2>UAV Webcam Streaming</h2>

   <video id="video" autoplay muted playsinline
       style="width: 640px; max-width: 100%;"></video>

    <p id="status">Connecting...</p>
    <p id="metrics">Waiting for video...</p>

    <script>
        const pc = new RTCPeerConnection();

        const video = document.getElementById("video");
        const status = document.getElementById("status");
        const metrics = document.getElementById("metrics");

        pc.addTransceiver("video", {
            direction: "recvonly"
        });

       pc.ontrack = async (event) => {
    video.srcObject = event.streams[0];
    video.muted = true;

    try {
        await video.play();
        status.innerText = "Video playing";
    } catch (error) {
        status.innerText = "Playback error: " + error.message;
        console.error(error);
    }
};
        async function startStream() {

            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);

            const response = await fetch("/offer", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    sdp: pc.localDescription.sdp,
                    type: pc.localDescription.type
                })
            });

            const answer = await response.json();

            await pc.setRemoteDescription(answer);
        }

        async function updateMetrics() {

            const stats = await pc.getStats();

            let fps = "N/A";
            let latency = "N/A";

            stats.forEach((report) => {

                if (report.type === "inbound-rtp" &&
                    report.kind === "video") {

                    if (report.framesPerSecond !== undefined) {
                        fps = report.framesPerSecond.toFixed(1);
                    }
                }

                if (report.type === "candidate-pair" &&
                    report.state === "succeeded" &&
                    report.currentRoundTripTime !== undefined) {

                    latency = (
                        report.currentRoundTripTime * 1000
                    ).toFixed(1) + " ms";
                }
            });

            metrics.innerText =
                "Video FPS: " + fps +
                " | Network RTT: " + latency;
        }

        setInterval(updateMetrics, 1000);

        startStream().catch((error) => {
            status.innerText = "Connection failed: " + error;
        });
    </script>
</body>
</html>
"""


async def index(request):
    return web.Response(text=HTML, content_type="text/html")


async def offer(request):

    params = await request.json()

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():

        print("Connection state:", pc.connectionState)

        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(
        RTCSessionDescription(
            sdp=params["sdp"],
            type=params["type"]
        )
    )

    pc.addTrack(WebcamStream())

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })


async def cleanup(app):

    await asyncio.gather(
        *(pc.close() for pc in pcs)
    )


app = web.Application()

app.router.add_get("/", index)
app.router.add_post("/offer", offer)

app.on_shutdown.append(cleanup)


if __name__ == "__main__":

    print("Starting UAV video streaming...")
    print("Open http://localhost:8080 in your browser")

    web.run_app(
        app,
        host="0.0.0.0",
        port=8080
    )
