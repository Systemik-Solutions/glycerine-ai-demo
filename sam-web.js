import {
    SamModel,
    AutoProcessor,
    RawImage,
    Tensor,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0-alpha.5";

let processor = null;
let model = null;
let imageInput = null;
let imageProcessed = null;
let imageEmbeddings = null;


async function createProcessor() {
    console.log("Loading model and processor");
    const model_id = "Xenova/slimsam-77-uniform";
    model = await SamModel.from_pretrained(model_id, {
        // dtype: "fp16", // or "fp32"
        // device: "webgpu",
        quantized: true,
        revision: "boxes"
    });
    processor = await AutoProcessor.from_pretrained(model_id);
    console.log("Model and processor loaded");
}

async function encode(url) {
    console.log("Loading image and computing embeddings");
    imageInput = await RawImage.fromURL(url);

    // Recompute image embeddings
    imageProcessed = await processor(imageInput);
    imageEmbeddings = await model.get_image_embeddings(imageProcessed);
    console.log("Image embeddings computed");
}

async function decode() {

    // Prepare inputs for decoding
    const reshaped = imageProcessed.reshaped_input_sizes[0];
    // const lastPoints = [{ position: [100, 100], label: 1 }];
    // const points = lastPoints
    //     .map((x) => [x.position[0] * reshaped[1], x.position[1] * reshaped[0]])
    //     .flat(Infinity);
    // const labels = lastPoints.map((x) => BigInt(x.label)).flat(Infinity);
    // console.log(points);
    // console.log(labels);

    const points = [(481/1024) * reshaped[1], (599/1252) * reshaped[0]];
    const labels = [1];
    const box = [(627/1024) * reshaped[1], (388/1252) * reshaped[0], (861/1024) * reshaped[1], (851/1252) * reshaped[0]]

    //const num_points = lastPoints.length;
    const num_points = 1;
    const input_points = new Tensor("float32", points, [1, 1, num_points, 2]);
    const input_labels = new Tensor("int64", labels, [1, 1, num_points]);
    const input_boxes = new Tensor("float32", box, [1, 1, 4]);

    // Generate the mask
    const { pred_masks, iou_scores } = await model({
        ...imageEmbeddings,
        input_boxes,
    });

    // Post-process the mask
    const masks = await processor.post_process_masks(
        pred_masks,
        imageProcessed.original_sizes,
        imageProcessed.reshaped_input_sizes,
    );

    const mask = RawImage.fromTensor(masks[0][0]);
    const scores = iou_scores.data;
    drawMask(mask, scores);
    const polygon = getPolygon(mask, scores);
    console.log(polygon);
    drawPolygon(polygon);
}

function drawMask(mask, scores) {
    const maskCanvas = document.getElementById("mask");
    const maskContext = maskCanvas.getContext("2d");
    // Allocate buffer for pixel data
    const imageData = maskContext.createImageData(
        maskCanvas.width,
        maskCanvas.height,
    );

    // Select best mask
    const numMasks = scores.length; // 3
    let bestIndex = 0;
    for (let i = 1; i < numMasks; ++i) {
        if (scores[i] > scores[bestIndex]) {
            bestIndex = i;
        }
    }
    console.log(`Segment score: ${scores[bestIndex].toFixed(2)}`);

    // Fill mask with colour
    const pixelData = imageData.data;
    for (let i = 0; i < pixelData.length; ++i) {
        if (mask.data[numMasks * i + bestIndex] === 1) {
            const offset = 4 * i;
            pixelData[offset] = 0; // red
            pixelData[offset + 1] = 114; // green
            pixelData[offset + 2] = 189; // blue
            pixelData[offset + 3] = 255; // alpha
        }
    }

    // Draw image data to context
    maskContext.putImageData(imageData, 0, 0);
}

function getPolygon(mask, scores) {
    const numMasks = scores.length;
    let bestIndex = 0;
    for (let i = 1; i < numMasks; ++i) {
        if (scores[i] > scores[bestIndex]) {
            bestIndex = i;
        }
    }

    const maskData = mask.data;
    const maskWidth = mask.width;
    const maskHeight = mask.height;
    const points = [];
    for (let y = 0; y < maskHeight; ++y) {
        for (let x = 0; x < maskWidth; ++x) {
            const offset = y * maskWidth + x;
            if (maskData[numMasks * offset + bestIndex] === 1) {
                points.push([x, y]);
            }
        }
    }
    const polygon = hull(points, 10);
    return polygon;
}

function drawPolygon(polygon) {
    const canvas = document.getElementById("mask");
    const ctx = canvas.getContext("2d");
    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (const [x, y] of polygon) {
        ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.stroke();
}

async function run() {
    const imageUrl = "https://iaw-image-server.ardc-hdcl-sia-iaw.cloud.edu.au/images/iiif/01HM54MGYC39W9MZ32YD4GJ59D/full/1024,/0/default.jpg";
    // Render image in canvas.
    const container = document.getElementById("image-container");
    const canvas = document.createElement("canvas");
    canvas.id = "mask";
    const ctx = canvas.getContext("2d");
    const img = new Image();
    img.src = imageUrl;
    img.onload = () => {
        container.style.width = img.width + "px";
        container.style.height = img.height + "px";
        canvas.width = img.width;
        canvas.height = img.height;
        container.appendChild(img);
        container.appendChild(canvas);
    };
    await createProcessor();
    await encode(imageUrl);
    await decode();
}

await run();


