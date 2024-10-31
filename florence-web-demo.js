import {
    Florence2ForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    RawImage,
    SamModel,
    Tensor,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0/dist/transformers.min.js";

function info(message) {
    const info = document.getElementById("infoPane");
    // Append message paragraph to info element.
    const p = document.createElement("p");
    p.textContent = message;
    info.appendChild(p);
    // Scroll to bottom of info element.
    info.scrollTop = info.scrollHeight;
}

function display_model_progress(progress) {
    let progressElement = document.getElementById("progress");
    if (!progressElement) {
        progressElement = document.createElement("div");
        progressElement.id = "progress";
        document.getElementById("infoPane").appendChild(progressElement);
    }
    const fileName = progress.file;
    let progressVal = progress.status === 'progress' ? progress.progress : 100;
    let fileEle = document.getElementById(fileName);
    if (!fileEle) {
        fileEle = document.createElement("div");
        fileEle.id = fileName;
        progressElement.appendChild(fileEle);
    }
    fileEle.innerHTML = `<p>${fileName}: <span>${Math.round(progressVal)}</span>%</p>`;
}

async function runFlorenceTask(task, textInput = null) {
    if (textInput) {
        task += textInput;
    }
    const prompts = fProcessor.construct_prompts(task);
    const text_inputs = fTokenizer(prompts);

    // Generate text
    const generated_ids = await fModel.generate({
        ...text_inputs,
        ...vision_inputs,
        max_new_tokens: 100,
    });

    // Decode generated text
    const generated_text = fTokenizer.batch_decode(generated_ids, { skip_special_tokens: false })[0];

    // Post-process the generated text
    const result = fProcessor.post_process_generation(generated_text, task, image.size);
    return result;
}

function boxToLoc(box) {
    // Convert the box to 0-999 range.
    const region = [
        Math.floor(box[0] / image.width * 999),
        Math.floor(box[1] / image.height * 999),
        Math.floor(box[2] / image.width * 999),
        Math.floor(box[3] / image.height * 999)
    ];
    return `<loc_${region[0]}><loc_${region[1]}><loc_${region[2]}><loc_${region[3]}>`;
}

/**
 * Convert the raw location string to a polygon.
 * 
 * @param {string} raw
 *   The raw location string in the format `<loc_px1><loc_py1><loc_px2><loc_py2...>`.
 * @returns {Array}
 *   The polygon in the format `[[x1, y1], [x2, y2], ...]`.
 */
function locToPolygon(raw) {
    const polygon = [];
    const locs = raw.match(/<loc_\d+>/g);
    const originalImageWidth = viewer.world.getItemAt(0).getContentSize().x;
    const originalImageHeight = viewer.world.getItemAt(0).getContentSize().y;
    for (let i = 0; i < locs.length; i += 2) {
        const x = parseInt(locs[i].slice(5, -1));
        const y = parseInt(locs[i + 1].slice(5, -1));
        polygon.push([x * originalImageWidth / 1000, y * originalImageHeight / 1000]);
    }
    return polygon;

}

async function denseRegionCaption() {
    const task = '<DENSE_REGION_CAPTION>';
    const result = await runFlorenceTask(task);
    return result[Object.keys(result)[0]];
}

async function regionToSegmentation(box) {
    const task = '<REGION_TO_SEGMENTATION>';
    const result = await runFlorenceTask(task, boxToLoc(box));
    return result[Object.keys(result)[0]];
}

async function regionProposal() {
    const task = '<REGION_PROPOSAL>';
    const result = await runFlorenceTask(task);
    return result[Object.keys(result)[0]];
}

async function regionToDescription(box) {
    const task = '<REGION_TO_DESCRIPTION>';
    const result = await runFlorenceTask(task, boxToLoc(box));
    let description = result[Object.keys(result)[0]];
    // Remove tags from the description
    description = description.replace(/<[^>]*>/g, '');
    return description;
}

async function drawPolygon(selection = null, polygon) {
    if (!selection) {
        selection = {
            "@context": "http://www.w3.org/ns/anno.jsonld",
            "type": "Annotation",
            "body": [],
            "target": {},
        }
    } else {
        anno.cancelSelected();
    }
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
    
    anno.addAnnotation(selection);
    return selection.id;
}

function getSelectionBoundingBox(selection) {
    // Get the bounding box of the selection
    const originalImageWidth = viewer.world.getItemAt(0).getContentSize().x;
    const originalImageHeight = viewer.world.getItemAt(0).getContentSize().y;
    const selectorValue = selection.target.selector.value;  // In "xywh=pixel:0,0,100,100" format.
    const xywh = selectorValue.split('=')[1].split(':').pop().split(',').map(Number);
    let boundingBox = xywh.map((x, i) => i < 2 ? x : x + xywh[i - 2]);
    // Scale the bounding box to the processed image size.
    boundingBox = boundingBox.map(x => Math.round(x * image.width / originalImageWidth));
    return boundingBox;
}

async function florenceSegment(selection) {
    const loader = document.getElementById("loaderPane");
    loader.style.display = "flex";
    info("Segmenting in bounding box...");
    const boundingBox = getSelectionBoundingBox(selection);
    const description = await regionToDescription(boundingBox);

    const rawLoc = await regionToSegmentation(boundingBox);
    const polygon = locToPolygon(rawLoc);
    const anoID = await drawPolygon(selection, polygon);
    annoDescriptions[anoID] = description;
    
    info(`Polygon (size ${polygon.length}) created`);
    loader.style.display = "none";
}

async function florenceSegmentAll() {
    const loader = document.getElementById("loaderPane");
    loader.style.display = "flex";
    info("Segmenting all regions...");
    const rp = await regionProposal();
    for (let bbox of rp.bboxes) {
        const rawLoc = await regionToSegmentation(bbox);
        const polygon = locToPolygon(rawLoc);
        const anoID = await drawPolygon(null, polygon);
        const description = await regionToDescription(bbox);
        annoDescriptions[anoID] = description;
    }
    info("All regions segmented");
    loader.style.display = "none";
}

async function samSegment(boundingBox) {
    const originalImageWidth = viewer.world.getItemAt(0).getContentSize().x;
    const originalImageHeight = viewer.world.getItemAt(0).getContentSize().y;
    
    // Prepare inputs for decoding
    const reshaped = imageProcessed.reshaped_input_sizes[0];

    const box = [
        (boundingBox[0] / image.width) * reshaped[1],
        (boundingBox[1] / image.height) * reshaped[0],
        (boundingBox[2] / image.width) * reshaped[1],
        (boundingBox[3] / image.height) * reshaped[0],
    ];

    const input_boxes = new Tensor("float32", box, [1, 1, 4]);

    // Generate the mask
    const { pred_masks, iou_scores } = await sModel({
        ...imageEmbeddings,
        input_boxes,
    });

    // Post-process the mask
    const masks = await sProcessor.post_process_masks(
        pred_masks,
        imageProcessed.original_sizes,
        imageProcessed.reshaped_input_sizes,
    );

    const mask = RawImage.fromTensor(masks[0][0]);
    const scores = iou_scores.data;
    const polygon = getPolygonFromMask(mask, scores);
    return polygon;
}

function getPolygonFromMask(mask, scores) {
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

    const removeNoise = document.getElementById("ckbRmNoise").checked;
    let refinedPoints = points;
    if (removeNoise) {
        // Cluster the mask points
        const dbscan = new DBSCAN();
        const clusters = dbscan.run(points, 5, 10);
        // Find the largest cluster
        let largestCluster = [];
        for (const cluster of clusters) {
            if (cluster.length > largestCluster.length) {
                largestCluster = cluster;
            }
        }
        refinedPoints = largestCluster.map(i => points[i]);
    }
    
    let polygon = hull(refinedPoints, 10);
    const originalImageWidth = viewer.world.getItemAt(0).getContentSize().x;
    const originalImageHeight = viewer.world.getItemAt(0).getContentSize().y;
    polygon = polygon.map(([x, y]) => [
        (x / image.width) * originalImageWidth,
        (y / image.height) * originalImageHeight,
    ]);
    return polygon;
}

async function combinedSegment(selection) {
    const loader = document.getElementById("loaderPane");
    loader.style.display = "flex";
    info("Segmenting in bounding box...");
    const boundingBox = getSelectionBoundingBox(selection);
    const description = await regionToDescription(boundingBox);

    const polygon = await samSegment(boundingBox);
    const anoID = await drawPolygon(selection, polygon);
    annoDescriptions[anoID] = description;
    
    info(`Polygon (size ${polygon.length}) created`);
    loader.style.display = "none";
}

async function combinedSegmentAll() {
    const loader = document.getElementById("loaderPane");
    loader.style.display = "flex";
    info("Segmenting all regions...");
    const rp = await regionProposal();
    for (let bbox of rp.bboxes) {
        const polygon = await samSegment(bbox);
        const anoID = await drawPolygon(null, polygon);
        const description = await regionToDescription(bbox);
        annoDescriptions[anoID] = description;
    }
    info("All regions segmented");
    loader.style.display = "none";
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
    'Gichi Nala': 'https://heidicon.ub.uni-heidelberg.de/iiif/2/23736792%3A960606',
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

const annoDescriptions = {};

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
    combinedSegment(selection);
});

const tooltip = document.createElement('div');
tooltip.className = 'anno-tooltip';
Object.assign(tooltip.style, {
    display: 'none',
    position: 'absolute',
    backgroundColor: 'white',
    color: '#595959',
    fontSize: '0.8rem',
    padding: '0.5em',
    boxShadow: '0 3px 10px rgb(0 0 0 / 0.2)',
    maxWidth: '300px'
});
document.body.appendChild(tooltip);

anno.on('mouseEnterAnnotation', function(annotation, element) {
    tooltip.textContent = annoDescriptions[annotation.id];
    let position = element.getBoundingClientRect();
    tooltip.style.top = `${position.top + 10}px`;
    tooltip.style.left = `${position.left + 10}px`;
    tooltip.style.display = 'block';
});

anno.on('mouseLeaveAnnotation', function(annotation, element) {
    tooltip.style.display = 'none';
    tooltip.innerHTML = '';
});

document.getElementById("segAll").addEventListener("click", combinedSegmentAll);

const loader = document.getElementById("loaderPane");
loader.style.display = "flex";

// Load model, processor, and tokenizer
info("Loading model...");

// Load florence-2 model
const f_model_id = 'onnx-community/Florence-2-base-ft';
// const model_id = 'onnx-community/Florence-2-large-ft';
const fModel = await Florence2ForConditionalGeneration.from_pretrained(f_model_id, { dtype: 'fp32', device: 'webgpu', progress_callback: (progress) => display_model_progress(progress) });
const fProcessor = await AutoProcessor.from_pretrained(f_model_id);
const fTokenizer = await AutoTokenizer.from_pretrained(f_model_id);

// Load SAM model
let sProcessor = null;
let sModel = null;
let imageProcessed = null;
let imageEmbeddings = null;

const model_id = "Xenova/slimsam-77-uniform";
sModel = await SamModel.from_pretrained(model_id, {
    dtype: "fp32", // or "fp32"
    device: "webgpu",
    quantized: true,
    revision: "boxes"
});
sProcessor = await AutoProcessor.from_pretrained(model_id);

info("Model loaded");

// Load and process image
info(`Loading image...`);
const processImageWidth = 1024;
const imageUrl = `${imageBase}/full/${processImageWidth},/0/default.jpg`;
const image = await RawImage.fromURL(imageUrl);

// Process the image in Florence
const vision_inputs = await fProcessor(image);

// Process the image in SAM
imageProcessed = await sProcessor(image);
imageEmbeddings = await sModel.get_image_embeddings(imageProcessed);

info(`Image loaded`);

loader.style.display = "none";
