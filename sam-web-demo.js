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

function info(message) {
    const info = document.getElementById("infoPane");
    // Append message paragraph to info element.
    const p = document.createElement("p");
    p.textContent = message;
    info.appendChild(p);
    // Scroll to bottom of info element.
    info.scrollTop = info.scrollHeight;
}

async function createProcessor() {
    info("Loading model and processor...");
    const model_id = "Xenova/slimsam-77-uniform";
    model = await SamModel.from_pretrained(model_id, {
        // dtype: "fp16", // or "fp32"
        // device: "webgpu",
        quantized: true,
        revision: "boxes"
    });
    processor = await AutoProcessor.from_pretrained(model_id);
    info("Model and processor loaded");
}

async function encode(url) {
    info("Loading image and computing embeddings...");
    imageInput = await RawImage.fromURL(url);

    // Recompute image embeddings
    imageProcessed = await processor(imageInput);
    imageEmbeddings = await model.get_image_embeddings(imageProcessed);
    info("Image embeddings computed");
}

async function decode(boundingBox) {
    const originalImageWidth = viewer.world.getItemAt(0).getContentSize().x;
    const originalImageHeight = viewer.world.getItemAt(0).getContentSize().y;
    
    // Prepare inputs for decoding
    const reshaped = imageProcessed.reshaped_input_sizes[0];

    //const box = [(627/1024) * reshaped[1], (388/1252) * reshaped[0], (861/1024) * reshaped[1], (851/1252) * reshaped[0]]

    const box = [
        (boundingBox[0] / originalImageWidth) * reshaped[1],
        (boundingBox[1] / originalImageHeight) * reshaped[0],
        (boundingBox[2] / originalImageWidth) * reshaped[1],
        (boundingBox[3] / originalImageHeight) * reshaped[0],
    ];

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
    const polygon = getPolygon(mask, scores);
    return polygon;
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
    let polygon = hull(points, 10);
    const originalImageWidth = viewer.world.getItemAt(0).getContentSize().x;
    polygon = polygon.map(([x, y]) => [
        (x / processImageWidth) * originalImageWidth,
        (y / processImageWidth) * originalImageWidth,
    ]);
    return polygon;
}

async function drawPolygon(selection, polygon) {
    const newSelector = {
        "type": "SvgSelector",
    };
    let svg = '<svg><polygon points="';
    for (const [x, y] of polygon) {
        svg += `${x},${y} `;
    }
    svg += '"/></svg>';
    newSelector.value = svg;
    selection.target.selector = newSelector;
    // Generate a random ID for the new annotation
    selection.id = Math.random().toString(36).substring(2, 15);
    anno.cancelSelected();
    anno.addAnnotation(selection);
}

async function segment(selection) {
    info("Segmenting in bounding box...");

    // Get the bounding box of the selection
    const selectorValue = selection.target.selector.value;  // In "xywh=pixel:0,0,100,100" format.
    const xywh = selectorValue.split('=')[1].split(':').pop().split(',').map(Number);
    const boundingBox = xywh.map((x, i) => i < 2 ? x : x + xywh[i - 2]);
    
    const polygon = await decode(boundingBox);
    await drawPolygon(selection, polygon);
    info(`Polygon (size ${polygon.length}) created`);
}

const imageList = {
    'Buddha Year 5': 'https://iaw-image-server.ardc-hdcl-sia-iaw.cloud.edu.au/images/iiif/01HM54MGYC39W9MZ32YD4GJ59D',
    'Schist Bodhisattva Head - Front': 'https://iaw-image-server.ardc-hdcl-sia-iaw.cloud.edu.au/images/iiif/01HGY0E51T8B85A0023YNHN2R9',
    'Shakyamuni and Maitreya': 'https://iaw-image-server.ardc-hdcl-sia-iaw.cloud.edu.au/images/iiif/01HGY2X02VBKQKW21J5AXKVT2Z',
    'Kanishka Casket': 'https://iaw-image-server.ardc-hdcl-sia-iaw.cloud.edu.au/images/iiif/01HM54F7AZ3NXRBR5C3TQHNGHR',
    'Inscribed Stele': 'https://iaw-image-server.ardc-hdcl-sia-iaw.cloud.edu.au/images/iiif/01HM54KWVM6DYFWGYZ5CECPKHN',
    'Bar Photo': 'https://iaw-image-server.ardc-hdcl-sia-iaw.cloud.edu.au/images/iiif/01HGWC2EGD0VV6QTT3X91V9Z1H',
    'Guanyin Statue': 'https://puam-loris.aws.princeton.edu/loris/y1950-66_DET.jp2',
    'IR Photo': 'https://iiif.io/api/image/3.0/example/reference/918ecd18c2592080851777620de9bcb5-gottingen',
    'John Dee': 'https://iiif.io/api/image/3.0/example/reference/421e65be2ce95439b3ad6ef1f2ab87a9-dee-natural',
    'Book Page': 'https://iiif.io/api/image/3.0/example/reference/59d09e6773341f28ea166e9f3c1e674f-gallica_ark_12148_bpt6k1526005v_f20',
    'Whistlers_Mother': 'https://iiif.io/api/image/3.0/example/reference/329817fc8a251a01c393f517d8a17d87-Whistlers_Mother',
    'Old train photo': 'https://images.prov.vic.gov.au/loris/2164%2F1297%2F05%2Fimages%2F1%2Ffiles%2F12800-00001-000170-280.tif',
};

let imageBase;

// Use the query string `image` as the image base URL. If not provided, use the default image.
const urlParams = new URLSearchParams(window.location.search);
const imageParam = urlParams.get('image');
if (imageParam) {
    imageBase = imageParam;
} else {
    imageBase = imageList['Buddha Year 5'];
}

const processImageWidth = 1024;
const imageUrl = `${imageBase}/full/${processImageWidth},/0/default.jpg`;

const viewer = OpenSeadragon({
    id: "osdContainer",
    visibilityRatio: 1,
    crossOriginPolicy: false,
    prefixUrl: "node_modules/openseadragon/build/openseadragon/images/",
    tileSources: `${imageBase}/info.json`,
});



const anno = OpenSeadragon.Annotorious(viewer, {
    allowEmpty: true,
    disableEditor: true,
});

anno.on('createSelection', function(selection) {
    segment(selection);
});

await createProcessor();
await encode(imageUrl);
info("Ready!");

